import pandas as pd
import pytest

from src.data.loader import load_emscad, combined_text, add_basic_flags
from helpers import make_df


def test_load_valid_file(tmp_path):
    p = tmp_path / "jobs.csv"
    make_df().to_csv(p, index=False)
    df = load_emscad(p)
    assert len(df) == 200
    assert set(df["fraudulent"].unique()) <= {0, 1}


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    make_df().drop(columns=["description"]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_emscad(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_emscad(tmp_path / "does_not_exist.csv")


def test_combined_text_handles_missing():
    df = make_df(10)
    text = combined_text(df)
    assert text.notna().all()
    assert "Posting number 0" in text.iloc[0]


def test_basic_flags():
    out = add_basic_flags(make_df(50))
    assert {"no_salary", "no_company_profile", "country"} <= set(out.columns)
    assert out["no_salary"].eq(1).all()          # salary_range is empty in the test data
