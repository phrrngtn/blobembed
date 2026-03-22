"""Packaging wrapper for the blobembed SQLite extension."""

import pathlib

_HERE = pathlib.Path(__file__).parent


def extension_path() -> str:
    """Return the absolute path to the blobembed SQLite extension (without suffix).

    SQLite's .load command does not want the file extension:
        .load <path>
    """
    base = _HERE / "blobembed"
    for suffix in (".so", ".dylib", ".dll"):
        if (base.parent / f"blobembed{suffix}").exists():
            return str(base)
    raise FileNotFoundError(f"Extension not found at {base}.*")
