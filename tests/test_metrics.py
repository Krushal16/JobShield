import numpy as np

from src.evaluation.metrics import evaluate


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1, 0])
    r = evaluate(y, y, y_score=y.astype(float))
    assert r["f1"] == 1 and r["precision"] == 1 and r["recall"] == 1


def test_always_real_baseline_has_high_accuracy_but_zero_recall():
    y = np.array([0] * 95 + [1] * 5)
    r = evaluate(y, np.zeros_like(y))
    assert r["accuracy"] == 0.95
    assert r["recall"] == 0 and r["f1"] == 0
    assert r["fn"] == 5 and r["tp"] == 0
