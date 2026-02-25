# utility functions used by pipeline.py and tested in tests.py

import re
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo

from springer_v2.config import NULL_VALUES


def convert_utc_to_local(ts, tz_str, fallback="Asia/Jakarta"):
    """Convert a UTC-aware timestamp to naive local time."""
    if pd.isna(ts):
        return pd.NaT
    tz = fallback if (not tz_str or pd.isna(tz_str)) else str(tz_str)
    try:
        return ts.astimezone(ZoneInfo(tz)).replace(tzinfo=None)
    except Exception:
        return ts.replace(tzinfo=None)


def extract_number_from_text(text):
    """
    Pull first number out of a string like '10 days' -> 10.
    Returns None if no number found.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def safe_boolean_conversion(series):
    """
    Convert string series to boolean.
    Handles: true/false, TRUE/FALSE, 1/0, yes/no
    Invalid values become pd.NA
    """
    truthy  = {"true", "1", "yes", "y"}
    falsy   = {"false", "0", "no", "n"}

    def convert(val):
        if pd.isna(val):
            return pd.NA
        v = str(val).strip().lower()
        if v in truthy:
            return True
        if v in falsy:
            return False
        return pd.NA

    return series.apply(convert)


def load_csv_with_nulls(path):
    """Load a CSV file treating common null strings as NaN."""
    return pd.read_csv(path, na_values=NULL_VALUES, keep_default_na=True)
