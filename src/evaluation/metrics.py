"""Shared evaluation functions (Week 2). Every model is scored the same way.

Main metrics are for the FRAUDULENT class (label 1): precision, recall, F1 and PR-AUC.
Accuracy is reported only for reference – with 4.8% fake postings, a model that always
predicts "real" already scores about 95% accuracy while catching zero scams.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src.utils.config import load_config, project_path

_CFG = load_config()
RESULTS_CSV = project_path(_CFG["paths"]["metrics_file"])
FIG_DIR = project_path(_CFG["paths"]["figures_dir"])


def evaluate(y_true, y_pred, y_score=None, model_name="model", split="val", notes=""):
    """Return a dict of metrics. y_score = predicted probability of 'fake' (optional)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    res = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_name, "split": split, "notes": notes,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "pr_auc": np.nan, "roc_auc": np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        res["pr_auc"] = average_precision_score(y_true, y_score)
        res["roc_auc"] = roc_auc_score(y_true, y_score)
    return res


def print_report(res):
    print(f"{res['model']} [{res['split']}]")
    print(f"  Precision {res['precision']:.3f} | Recall {res['recall']:.3f} | F1 {res['f1']:.3f} | "
          f"PR-AUC {res['pr_auc']:.3f} | Accuracy {res['accuracy']:.3f}")
    print(f"  Caught {res['tp']} of {res['tp'] + res['fn']} fake postings; "
          f"{res['fp']} real postings wrongly flagged")


def log_result(res, path=RESULTS_CSV):
    """Append one result row to results/metrics/results.csv (the project's model leaderboard)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([res])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def plot_confusion(res, save_as=None):
    cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred real", "Pred fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual real", "Actual fake"])
    ax.set_title(f"{res['model']} – {res['split']}")
    plt.tight_layout()
    if save_as:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / save_as, dpi=200)
    return fig


def plot_pr_curve(y_true, y_score, label="model", ax=None, save_as=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.plot(recall, precision, label=f"{label} (AP={average_precision_score(y_true, y_score):.2f})")
    ax.axhline(np.mean(y_true), ls="--", color="grey", lw=1, label="random")
    ax.set_xlabel("Recall (share of scams caught)"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Precision–recall curve")
    plt.tight_layout()
    if save_as:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIG_DIR / save_as, dpi=200)
    return ax
