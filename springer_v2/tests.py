"""
Unit tests for the Springer Capital Referral Pipeline.

This module contains tests for key functions to ensure code quality
and demonstrate testing practices for portfolio purposes.
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import our modules
from springer_v2.utils import convert_utc_to_local, extract_number_from_text, safe_boolean_conversion
from springer_v2.config import NULL_VALUES


class TestUtils:
    """Test utility functions."""

    def test_convert_utc_to_local(self):
        """Test timezone conversion functionality."""
        # Create a UTC timestamp
        utc_ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")

        # Convert to Jakarta timezone
        local_ts = convert_utc_to_local(utc_ts, "Asia/Jakarta")

        # Jakarta is UTC+7, so 12:00 UTC should be 19:00 local
        assert local_ts.hour == 19
        assert local_ts.day == 1
        assert local_ts.tz is None  # Should be naive

    def test_extract_number_from_text(self):
        """Test number extraction from text."""
        assert extract_number_from_text("10 days") == 10
        assert extract_number_from_text("Free for 30 days") == 30
        assert extract_number_from_text("No numbers here") is None
        assert extract_number_from_text("") is None

    def test_safe_boolean_conversion(self):
        """Test boolean conversion with various inputs."""
        series = pd.Series(["true", "false", "TRUE", "FALSE", "1", "0", "yes", "no"])

        result = safe_boolean_conversion(series)

        expected = pd.Series([True, False, True, False, True, False, True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_boolean_conversion_invalid(self):
        """Test boolean conversion with invalid values."""
        series = pd.Series(["maybe", "perhaps"])

        result = safe_boolean_conversion(series)

        # Invalid values should become NA
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])


class TestConfig:
    """Test configuration constants."""

    def test_null_values_defined(self):
        """Test that NULL_VALUES is properly defined."""
        assert isinstance(NULL_VALUES, list)
        assert "null" in NULL_VALUES
        assert "NULL" in NULL_VALUES
        assert "" in NULL_VALUES

    def test_paths_exist(self):
        """Test that configured paths are properly set."""
        from springer_v2.config import DATA_DIR, OUTPUT_DIR, PROFILE_DIR

        assert isinstance(DATA_DIR, Path)
        assert isinstance(OUTPUT_DIR, Path)
        assert isinstance(PROFILE_DIR, Path)


class TestDataLoading:
    """Test data loading functionality."""

    def test_csv_loading_with_nulls(self):
        """Test that CSV loading handles null values correctly."""
        from springer_v2.utils import load_csv_with_nulls

        # Create a small test CSV in memory
        test_data = "col1,col2,col3\nval1,null,normal\n,val2,another\n"
        test_file = Path("test_temp.csv")

        try:
            with open(test_file, 'w') as f:
                f.write(test_data)

            df = load_csv_with_nulls(test_file)

            # Check that null values are properly handled
            assert pd.isna(df.loc[0, 'col2'])  # "null" string should be NaN
            assert pd.isna(df.loc[1, 'col1'])  # Empty string should be NaN
            assert df.loc[0, 'col1'] == 'val1'
            assert df.loc[1, 'col2'] == 'val2'

        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])