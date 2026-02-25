# config.py - paths and shared constants for the pipeline

from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "output"
PROFILE_DIR = OUTPUT_DIR / "data_profiling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# values to treat as null when reading CSVs
NULL_VALUES = ["null", "NULL", "None", "none", "NA", "N/A", ""]
