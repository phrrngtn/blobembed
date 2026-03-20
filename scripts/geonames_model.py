"""
SQLAlchemy model for GeoNames gazetteer data.

Tables:
  - geonames_country: country definitions
  - geonames_admin1: first-level admin divisions
  - geonames_admin2: second-level admin divisions
  - geonames_place: populated places with joined hierarchy

Usage:
  uv run python scripts/geonames_model.py create   # create schema in all dialects
  uv run python scripts/geonames_model.py export    # export from DuckDB, import to all
"""

import sys

from sqlalchemy import (
    Column, Float, Index, Integer, MetaData, String, Table, Text,
    create_engine,
)

# No schema in MetaData — we set search_path/default schema at the engine level
# for PG. SQLite doesn't support schemas.
metadata = MetaData()

geonames_country = Table(
    "geonames_country", metadata,
    Column("iso", String(2), primary_key=True),
    Column("iso3", String(3), nullable=False),
    Column("iso_numeric", String(3)),
    Column("country", String(200), nullable=False),
    Column("capital", String(200)),
    Column("continent", String(2)),
    Column("currency_code", String(3)),
    Column("currency_name", String(100)),
    Column("languages", String(200)),
    Column("geonameid", Integer),
    Column("neighbours", String(200)),
)

geonames_admin1 = Table(
    "geonames_admin1", metadata,
    Column("code", String(40), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("asciiname", String(200)),
    Column("geonameid", Integer),
)

geonames_admin2 = Table(
    "geonames_admin2", metadata,
    Column("code", String(80), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("asciiname", String(200)),
    Column("geonameid", Integer),
)

geonames_place = Table(
    "geonames_place", metadata,
    Column("geonameid", Integer, primary_key=True),
    Column("place_name", String(200)),
    Column("place_ascii", String(200)),
    # alternatenames dropped — too verbose, marginal value
    Column("latitude", Float),
    Column("longitude", Float),
    Column("feature_code", String(10)),
    Column("population", Integer),
    Column("country_code", String(2)),
    Column("country_name", String(200)),
    Column("country_iso3", String(3)),
    Column("continent", String(2)),
    Column("admin1_name", String(200)),
    Column("admin2_name", String(200)),
    Column("full_path", Text),

    Index("idx_place_country", "country_code"),
    Index("idx_place_admin1", "country_code", "admin1_name"),
    Index("idx_place_name", "place_ascii"),
    Index("idx_place_population", "population"),
)


ENGINES = {
    "sqlite": "sqlite:///data/geonames/gazetteer.sqlite",
    "postgresql": "postgresql://rule4@localhost/rule4_test?options=-csearch_path%3Dgazetteer,public",
    # SQL Server via pyodbc — adjust DSN as needed
    # "mssql": "mssql+pyodbc://rule4_test_dsn",
}


def create_schemas():
    """Create the gazetteer schema in all configured databases."""
    for name, url in ENGINES.items():
        print(f"Creating schema in {name}...")
        engine = create_engine(url)
        if name in ("postgresql", "mssql"):
            with engine.connect() as conn:
                conn.execute(
                    __import__("sqlalchemy").text(
                        "CREATE SCHEMA IF NOT EXISTS gazetteer"
                    )
                )
                conn.commit()
        metadata.create_all(engine)
        print(f"  done.")


def export_and_import():
    """Export from DuckDB parquet, import to all dialects via SQLAlchemy."""
    import duckdb
    import csv
    import io

    # Export from DuckDB to CSV in memory
    print("Exporting from DuckDB...")
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs")

    # Rerun the load script to get data into DuckDB
    with open("sql/geonames_load.sql") as f:
        for statement in f.read().split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith(".print") and not stmt.startswith("INSTALL") and not stmt.startswith("LOAD"):
                try:
                    con.execute(stmt)
                except Exception as e:
                    # Skip DuckDB-only syntax (.print etc)
                    if "Parser Error" not in str(e):
                        print(f"  warning: {e}")

    # Fetch data as Python lists
    countries = con.execute("SELECT iso, iso3, iso_numeric, country, capital, continent, currency_code, currency_name, languages, geonameid::INTEGER, neighbours FROM geonames_countries").fetchall()
    admin1 = con.execute("SELECT code, name, asciiname, geonameid FROM geonames_admin1").fetchall()
    admin2 = con.execute("SELECT code, name, asciiname, geonameid FROM geonames_admin2").fetchall()
    places = con.execute("SELECT geonameid, place_name, place_ascii, alternatenames, latitude, longitude, feature_code, population, country_code, country_name, country_iso3, continent, admin1_name, admin2_name, full_path FROM geonames_places").fetchall()

    print(f"  countries: {len(countries)}, admin1: {len(admin1)}, admin2: {len(admin2)}, places: {len(places)}")
    con.close()

    # Import to each engine
    for name, url in ENGINES.items():
        print(f"Importing to {name}...")
        engine = create_engine(url)

        with engine.begin() as conn:
            # Clear existing data (order matters for FKs)
            conn.execute(geonames_place.delete())
            conn.execute(geonames_admin2.delete())
            conn.execute(geonames_admin1.delete())
            conn.execute(geonames_country.delete())

            # Insert in batches
            if countries:
                conn.execute(geonames_country.insert(), [
                    dict(zip(["iso", "iso3", "iso_numeric", "country", "capital",
                              "continent", "currency_code", "currency_name",
                              "languages", "geonameid", "neighbours"], row))
                    for row in countries
                ])

            if admin1:
                conn.execute(geonames_admin1.insert(), [
                    dict(zip(["code", "name", "asciiname", "geonameid"], row))
                    for row in admin1
                ])

            if admin2:
                conn.execute(geonames_admin2.insert(), [
                    dict(zip(["code", "name", "asciiname", "geonameid"], row))
                    for row in admin2
                ])

            # Places in batches of 10000
            place_cols = ["geonameid", "place_name", "place_ascii", "alternatenames",
                          "latitude", "longitude", "feature_code", "population",
                          "country_code", "country_name", "country_iso3", "continent",
                          "admin1_name", "admin2_name", "full_path"]
            for i in range(0, len(places), 10000):
                batch = places[i:i+10000]
                conn.execute(geonames_place.insert(), [
                    dict(zip(place_cols, row)) for row in batch
                ])
                print(f"  places: {min(i+10000, len(places))}/{len(places)}")

        print(f"  done.")


def main():
    if len(sys.argv) < 2:
        print("usage: geonames_model.py [create|export]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "create":
        create_schemas()
    elif cmd == "export":
        create_schemas()  # ensure tables exist
        export_and_import()
    else:
        print(f"unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
