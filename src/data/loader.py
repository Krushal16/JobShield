"""Load and validate the EMSCAD fake job postings dataset.

Usage:
    from src.data.loader import load_emscad
    df = load_emscad()                      # reads the raw_data path in configs/config.yaml
"""
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, project_path

DEFAULT_PATH = project_path(load_config()["paths"]["raw_data"])

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLS = ["location", "department", "salary_range", "employment_type",
                    "required_experience", "required_education", "industry", "function"]
BINARY_COLS = ["telecommuting", "has_company_logo", "has_questions"]
TARGET = "fraudulent"
EXPECTED_COLS = ["job_id"] + TEXT_COLS[:1] + ["location", "department", "salary_range"] + \
    TEXT_COLS[1:] + BINARY_COLS + ["employment_type", "required_experience",
                                   "required_education", "industry", "function", TARGET]


def load_emscad(path=DEFAULT_PATH, validate=True):
    """Read the CSV and (optionally) check that it matches the expected schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download fake_job_postings.csv from "
            "https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction "
            "and place it in data/raw/ (see README, section Dataset).")
    df = pd.read_csv(path)
    if validate:
        missing = set(EXPECTED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")
        bad = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(f"Unexpected target values: {bad}")
    return df


def combined_text(df):
    """Merge all text fields into one string per posting (used by the models)."""
    return df[TEXT_COLS].fillna("").agg(" ".join, axis=1).str.strip()


def add_basic_flags(df):
    """Simple red-flag features used in EDA and later in the XGBoost model."""
    out = df.copy()
    out["no_salary"] = out["salary_range"].isna().astype(int)
    out["no_company_profile"] = out["company_profile"].isna().astype(int)
    out["country"] = out["location"].fillna("Unknown").str.split(",").str[0].str.strip()
    out.loc[out["country"] == "", "country"] = "Unknown"
    return out
