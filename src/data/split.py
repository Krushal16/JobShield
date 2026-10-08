"""Stratified 70/15/15 train / validation / test split (Week 2).

Every model in the project uses these same three files, so results are comparable.
The test set is not used until final model comparison (Week 7).
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import TARGET
from src.utils.config import load_config, project_path

_CFG = load_config()
SEED = _CFG["seed"]
SPLIT_DIR = project_path(_CFG["paths"]["processed_dir"])


def make_splits(df, train=None, val=None, test=None, seed=SEED):
    """Split while keeping the same fake/real ratio in every part (ratios default to the config)."""
    train = _CFG["split"]["train"] if train is None else train
    val = _CFG["split"]["val"] if val is None else val
    test = _CFG["split"]["test"] if test is None else test
    assert abs(train + val + test - 1) < 1e-9, "proportions must sum to 1"
    train_df, temp_df = train_test_split(df, test_size=val + test, stratify=df[TARGET], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=test / (val + test),
                                       stratify=temp_df[TARGET], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_summary(parts):
    """Table of rows, fake count and fake % for each split."""
    rows = []
    for name, part in parts.items():
        rows.append({"split": name, "rows": len(part), "fake": int(part[TARGET].sum()),
                     "fake_%": round(part[TARGET].mean() * 100, 2)})
    return pd.DataFrame(rows)


def save_splits(train_df, val_df, test_df, out_dir=SPLIT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Saved train/val/test CSVs to {out_dir}")


def load_splits(out_dir=SPLIT_DIR):
    out_dir = Path(out_dir)
    return tuple(pd.read_csv(out_dir / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test"))
