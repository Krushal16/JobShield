"""Small synthetic EMSCAD-like dataframe used by the tests (no real data needed)."""
import numpy as np
import pandas as pd

from src.data.loader import EXPECTED_COLS


def make_df(n=200, fake_share=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: int(n * fake_share)] = 1
    rng.shuffle(y)
    df = pd.DataFrame({c: [None] * n for c in EXPECTED_COLS})
    df["job_id"] = np.arange(1, n + 1)
    df["title"] = [f"Job title {i}" for i in range(n)]
    df["location"] = rng.choice(["US, NY, New York", "GB, LND, London", None], n)
    df["description"] = [f"Posting number {i} with unique words alpha{i} beta{i}" for i in range(n)]
    df["company_profile"] = np.where(y == 1, None, "We are a real company")
    df["telecommuting"] = rng.integers(0, 2, n)
    df["has_company_logo"] = 1 - y
    df["has_questions"] = rng.integers(0, 2, n)
    df["employment_type"] = rng.choice(["Full-time", "Part-time", None], n)
    df["fraudulent"] = y
    return df
