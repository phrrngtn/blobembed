"""
Harvest domain enumerations from Wikidata SPARQL + hand-curated US federal
standards. Outputs YAML for loading into DuckDB via blobtemplates.

Wikidata domains: countries, US states, currencies, languages, chemical elements
Hand-curated: US Census race/ethnicity, school types, land use, crime categories,
              months, days, quarters, compass, gender, boolean labels

Usage:
    uv run python /tmp/harvest_domains.py > data/domain_enumerations.yaml
"""

import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "blobembed-domain-harvester/0.1 (https://github.com/phrrngtn/blobembed)"


def sparql_query(query):
    """Execute a SPARQL query against Wikidata and return results."""
    url = WIKIDATA_ENDPOINT + "?" + urllib.parse.urlencode({
        "query": query, "format": "json"
    })
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["results"]["bindings"]


def extract_labels(bindings, label_key="label", alt_key="altLabels"):
    """Extract primary label + alt labels from SPARQL bindings."""
    members = []
    for b in bindings:
        label = b[label_key]["value"]
        alts = b.get(alt_key, {}).get("value", "")
        alt_list = [a.strip() for a in alts.split(",") if a.strip()] if alts else []
        members.append({"label": label, "alt_labels": alt_list})
    return members


def yaml_escape(s):
    if not s:
        return '""'
    needs_quoting = any(ch in s for ch in (':', '#', '{', '}', '[', ']', ',', '&',
                                            '*', '?', '|', '-', '<', '>', '=', '!',
                                            '%', '@', '\\', '"', "'", '\n'))
    if s.strip() != s:
        needs_quoting = True
    lower = s.lower()
    if lower in ('true', 'false', 'yes', 'no', 'null', 'on', 'off'):
        needs_quoting = True
    if needs_quoting:
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
    try:
        float(s)
        return f'"{s}"'
    except ValueError:
        pass
    return s


# ── Wikidata queries ─────────────────────────────────────────────────

WIKIDATA_DOMAINS = [
    {
        "domain_name": "countries",
        "domain_label": "country",
        "source": "wikidata",
        "wikidata_qid": "Q6256",
        "query": """
            SELECT ?label (GROUP_CONCAT(DISTINCT ?altLabel; separator=", ") AS ?altLabels)
            WHERE {
              ?item wdt:P31 wd:Q6256.
              ?item rdfs:label ?label. FILTER(LANG(?label) = "en")
              OPTIONAL { ?item skos:altLabel ?altLabel. FILTER(LANG(?altLabel) = "en") }
            }
            GROUP BY ?label
            ORDER BY ?label
        """
    },
    {
        "domain_name": "us_states",
        "domain_label": "us_state",
        "source": "wikidata",
        "wikidata_qid": "Q35657",
        "query": """
            SELECT ?label ?code
            (GROUP_CONCAT(DISTINCT ?altLabel; separator=", ") AS ?altLabels)
            WHERE {
              ?item wdt:P31 wd:Q35657.
              ?item rdfs:label ?label. FILTER(LANG(?label) = "en")
              OPTIONAL { ?item wdt:P5087 ?code }
              OPTIONAL { ?item skos:altLabel ?altLabel. FILTER(LANG(?altLabel) = "en") }
            }
            GROUP BY ?label ?code
            ORDER BY ?label
        """
    },
    {
        "domain_name": "currencies",
        "domain_label": "currency",
        "source": "wikidata",
        "wikidata_qid": "Q8142",
        "query": """
            SELECT ?label ?code
            (GROUP_CONCAT(DISTINCT ?altLabel; separator=", ") AS ?altLabels)
            WHERE {
              ?item wdt:P31 wd:Q8142.
              ?item wdt:P498 ?code.
              ?item rdfs:label ?label. FILTER(LANG(?label) = "en")
              OPTIONAL { ?item skos:altLabel ?altLabel. FILTER(LANG(?altLabel) = "en") }
            }
            GROUP BY ?label ?code
            ORDER BY ?code
        """
    },
    {
        "domain_name": "languages_major",
        "domain_label": "language",
        "source": "wikidata",
        "wikidata_qid": "Q34770",
        "query": """
            SELECT ?label ?code
            (GROUP_CONCAT(DISTINCT ?altLabel; separator=", ") AS ?altLabels)
            WHERE {
              ?item wdt:P31 wd:Q34770.
              ?item wdt:P218 ?code.
              ?item wdt:P1098 ?speakers.
              ?item rdfs:label ?label. FILTER(LANG(?label) = "en")
              OPTIONAL { ?item skos:altLabel ?altLabel. FILTER(LANG(?altLabel) = "en") }
              FILTER(?speakers > 1000000)
            }
            GROUP BY ?label ?code
            ORDER BY ?label
        """
    },
    {
        "domain_name": "chemical_elements",
        "domain_label": "chemical_element",
        "source": "wikidata",
        "wikidata_qid": "Q11344",
        "query": """
            SELECT ?label ?symbol
            WHERE {
              ?item wdt:P31 wd:Q11344.
              ?item wdt:P246 ?symbol.
              ?item rdfs:label ?label. FILTER(LANG(?label) = "en")
            }
            ORDER BY ?label
        """
    },
]

# ── Hand-curated US federal standards ────────────────────────────────

CURATED_DOMAINS = [
    {
        "domain_name": "months_long",
        "domain_label": "month",
        "source": "curated",
        "members": [
            {"label": m, "alt_labels": []}
            for m in ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        ]
    },
    {
        "domain_name": "months_short",
        "domain_label": "month",
        "source": "curated",
        "members": [
            {"label": m, "alt_labels": []}
            for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ]
    },
    {
        "domain_name": "days_long",
        "domain_label": "day_of_week",
        "source": "curated",
        "members": [
            {"label": d, "alt_labels": []}
            for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ]
    },
    {
        "domain_name": "days_short",
        "domain_label": "day_of_week",
        "source": "curated",
        "members": [
            {"label": d, "alt_labels": []}
            for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        ]
    },
    {
        "domain_name": "quarters",
        "domain_label": "quarter",
        "source": "curated",
        "members": [
            {"label": q, "alt_labels": []}
            for q in ["Q1", "Q2", "Q3", "Q4", "Qtr1", "Qtr2", "Qtr3", "Qtr4"]
        ]
    },
    {
        "domain_name": "us_census_race",
        "domain_label": "race_ethnicity",
        "source": "curated:us_census",
        "members": [
            {"label": "White", "alt_labels": ["Caucasian"]},
            {"label": "Black or African American", "alt_labels": ["Black", "African American"]},
            {"label": "American Indian or Alaska Native", "alt_labels": ["Native American", "American Indian", "Alaska Native", "Indigenous"]},
            {"label": "Asian", "alt_labels": []},
            {"label": "Native Hawaiian or Other Pacific Islander", "alt_labels": ["Pacific Islander", "Native Hawaiian", "NHPI"]},
            {"label": "Two or More Races", "alt_labels": ["Multiracial", "Multi-racial", "Two or more"]},
            {"label": "Hispanic or Latino", "alt_labels": ["Hispanic", "Latino", "Latina", "Latinx"]},
            {"label": "Not Hispanic or Latino", "alt_labels": ["Non-Hispanic"]},
            {"label": "Some Other Race", "alt_labels": ["Other"]},
        ]
    },
    {
        "domain_name": "gender",
        "domain_label": "gender",
        "source": "curated",
        "members": [
            {"label": "Male", "alt_labels": ["M", "Man", "Men", "Boy"]},
            {"label": "Female", "alt_labels": ["F", "Woman", "Women", "Girl"]},
            {"label": "Non-binary", "alt_labels": ["Nonbinary", "NB", "Genderqueer"]},
            {"label": "Other", "alt_labels": ["X"]},
            {"label": "Unknown", "alt_labels": ["Not reported", "Prefer not to say"]},
        ]
    },
    {
        "domain_name": "boolean_labels",
        "domain_label": "boolean",
        "source": "curated",
        "members": [
            {"label": l, "alt_labels": []}
            for l in ["Yes", "No", "True", "False", "Y", "N", "T", "F",
                      "On", "Off", "Active", "Inactive", "Enabled", "Disabled"]
        ]
    },
    {
        "domain_name": "compass",
        "domain_label": "direction",
        "source": "curated",
        "members": [
            {"label": d, "alt_labels": []}
            for d in ["North", "South", "East", "West", "NE", "NW", "SE", "SW",
                      "Northeast", "Northwest", "Southeast", "Southwest"]
        ]
    },
    {
        "domain_name": "school_types",
        "domain_label": "school_type",
        "source": "curated:us_education",
        "members": [
            {"label": "Elementary School", "alt_labels": ["Elementary", "Primary", "Grade School"]},
            {"label": "Middle School", "alt_labels": ["Junior High", "Intermediate"]},
            {"label": "High School", "alt_labels": ["Secondary", "Senior High"]},
            {"label": "Charter School", "alt_labels": ["Charter"]},
            {"label": "Magnet School", "alt_labels": ["Magnet"]},
            {"label": "Vocational School", "alt_labels": ["Vocational", "Career", "Technical", "CTE"]},
            {"label": "Special Education", "alt_labels": ["SPED", "Special Ed"]},
            {"label": "Pre-K", "alt_labels": ["Pre-Kindergarten", "Preschool", "Early Childhood"]},
            {"label": "K-8", "alt_labels": ["K-8 School"]},
            {"label": "K-12", "alt_labels": ["K-12 School"]},
        ]
    },
    {
        "domain_name": "land_use",
        "domain_label": "land_use",
        "source": "curated:zoning",
        "members": [
            {"label": "Residential", "alt_labels": ["R", "RES"]},
            {"label": "Commercial", "alt_labels": ["C", "COM"]},
            {"label": "Industrial", "alt_labels": ["I", "IND", "Manufacturing"]},
            {"label": "Agricultural", "alt_labels": ["A", "AG", "Farm"]},
            {"label": "Institutional", "alt_labels": ["INST", "Government"]},
            {"label": "Mixed Use", "alt_labels": ["MU", "MXD"]},
            {"label": "Open Space", "alt_labels": ["OS", "Park", "Recreation"]},
            {"label": "Vacant", "alt_labels": ["VAC", "Undeveloped"]},
            {"label": "Transportation", "alt_labels": ["TRANS", "ROW", "Right of Way"]},
        ]
    },
    {
        "domain_name": "crime_categories",
        "domain_label": "crime",
        "source": "curated:fbi_ucr",
        "members": [
            {"label": "Homicide", "alt_labels": ["Murder", "Manslaughter"]},
            {"label": "Rape", "alt_labels": ["Sexual Assault"]},
            {"label": "Robbery", "alt_labels": []},
            {"label": "Aggravated Assault", "alt_labels": ["Assault", "Felony Assault"]},
            {"label": "Burglary", "alt_labels": ["Breaking and Entering", "B&E"]},
            {"label": "Larceny-Theft", "alt_labels": ["Larceny", "Theft", "Petit Larceny", "Grand Larceny"]},
            {"label": "Motor Vehicle Theft", "alt_labels": ["Auto Theft", "Car Theft", "GTA"]},
            {"label": "Arson", "alt_labels": []},
            {"label": "Drug/Narcotic Violations", "alt_labels": ["Drug", "Narcotics", "Controlled Substance"]},
            {"label": "DUI", "alt_labels": ["DWI", "Driving Under Influence"]},
            {"label": "Fraud", "alt_labels": ["Forgery", "Counterfeiting"]},
            {"label": "Vandalism", "alt_labels": ["Criminal Mischief", "Malicious Mischief"]},
            {"label": "Weapons Violation", "alt_labels": ["Weapons", "Firearms"]},
            {"label": "Prostitution", "alt_labels": ["Solicitation"]},
            {"label": "Disorderly Conduct", "alt_labels": ["Disturbing the Peace"]},
            {"label": "Trespass", "alt_labels": ["Criminal Trespass"]},
            {"label": "Domestic Violence", "alt_labels": ["DV", "Intimate Partner Violence"]},
        ]
    },
    {
        "domain_name": "utility_types",
        "domain_label": "utility",
        "source": "curated",
        "members": [
            {"label": "Water", "alt_labels": ["H2O", "Water Supply"]},
            {"label": "Sewer", "alt_labels": ["Wastewater", "Sanitary"]},
            {"label": "Electric", "alt_labels": ["Electricity", "Power", "Electrical"]},
            {"label": "Gas", "alt_labels": ["Natural Gas"]},
            {"label": "Stormwater", "alt_labels": ["Storm Drain", "Storm Sewer"]},
            {"label": "Telecommunications", "alt_labels": ["Telecom", "Cable", "Fiber"]},
            {"label": "Solid Waste", "alt_labels": ["Trash", "Garbage", "Refuse", "Waste"]},
        ]
    },
]


def fetch_wikidata_domain(domain):
    """Fetch members for a Wikidata-sourced domain."""
    print(f"  Fetching {domain['domain_name']} from Wikidata...", file=sys.stderr)
    bindings = sparql_query(domain["query"])
    members = []
    for b in bindings:
        label = b["label"]["value"]
        alts = b.get("altLabels", {}).get("value", "")
        alt_list = [a.strip() for a in alts.split(",") if a.strip()] if alts else []
        # Add code (ISO, symbol, etc.) as alt label if present
        for code_key in ("code", "symbol"):
            if code_key in b:
                code = b[code_key]["value"]
                if code and code not in alt_list:
                    alt_list.insert(0, code)
        members.append({"label": label, "alt_labels": alt_list})
    return members


def emit_yaml(domains):
    """Write all domains as YAML to stdout."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_wikidata = len([d for d in domains if d.get("source", "").startswith("wikidata")])
    n_curated = len(domains) - n_wikidata
    total_members = sum(len(d["members"]) for d in domains)

    print("metadata:")
    print(f'  extraction_date: "{now}"')
    print(f"  total_domains: {len(domains)}")
    print(f"  wikidata_domains: {n_wikidata}")
    print(f"  curated_domains: {n_curated}")
    print(f"  total_members: {total_members}")
    print('  description: "Domain enumerations for column name classification via blobfilters"')
    print()
    print("domains:")

    for d in domains:
        print(f"  - domain_name: {yaml_escape(d['domain_name'])}")
        print(f"    domain_label: {yaml_escape(d['domain_label'])}")
        print(f"    source: {yaml_escape(d.get('source', 'curated'))}")
        if "wikidata_qid" in d:
            print(f"    wikidata_qid: {yaml_escape(d['wikidata_qid'])}")
        print(f"    member_count: {len(d['members'])}")
        print(f"    members:")
        for m in d["members"]:
            print(f"      - label: {yaml_escape(m['label'])}")
            if m["alt_labels"]:
                alts_yaml = ", ".join(yaml_escape(a) for a in m["alt_labels"])
                print(f"        alt_labels: [{alts_yaml}]")


def main():
    all_domains = []

    # Fetch from Wikidata
    print("Fetching from Wikidata...", file=sys.stderr)
    for wd in WIKIDATA_DOMAINS:
        members = fetch_wikidata_domain(wd)
        wd["members"] = members
        all_domains.append(wd)
        print(f"    → {len(members)} members", file=sys.stderr)

    # Add curated domains
    print(f"Adding {len(CURATED_DOMAINS)} curated domains...", file=sys.stderr)
    all_domains.extend(CURATED_DOMAINS)

    total = sum(len(d["members"]) for d in all_domains)
    print(f"Total: {len(all_domains)} domains, {total} members", file=sys.stderr)

    emit_yaml(all_domains)


if __name__ == "__main__":
    main()
