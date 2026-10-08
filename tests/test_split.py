from src.data.split import make_splits
from helpers import make_df


def test_split_sizes_and_stratification():
    df = make_df(1000, fake_share=0.05)
    tr, va, te = make_splits(df, 0.70, 0.15, 0.15, seed=42)
    assert len(tr) + len(va) + len(te) == 1000
    assert abs(len(tr) - 700) <= 1 and abs(len(va) - 150) <= 1
    for part in (tr, va, te):
        assert abs(part["fraudulent"].mean() - 0.05) < 0.01


def test_no_overlap_between_splits():
    tr, va, te = make_splits(make_df(500), seed=42)
    a, b, c = set(tr.job_id), set(va.job_id), set(te.job_id)
    assert not (a & b) and not (a & c) and not (b & c)


def test_split_is_reproducible():
    df = make_df(300)
    first = make_splits(df, seed=42)[2].job_id.tolist()
    second = make_splits(df, seed=42)[2].job_id.tolist()
    assert first == second
