"""Week 3: tune the TF-IDF settings with stratified 5-fold cross-validation on the TRAINING set only.

    python -m src.training.tune_tfidf
"""
from itertools import product

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.loader import TARGET
from src.data.split import load_splits
from src.models.baseline import build_logreg_pipeline
from src.utils.config import load_config, project_path


def tune_tfidf(train_df, cfg=None, grid=None, n_jobs=-1, verbose=True, save=True):
    """Try every combination in cfg['tfidf_grid'] with Logistic Regression; return a table sorted by PR-AUC."""
    cfg = cfg or load_config()
    grid = grid or cfg["tfidf_grid"]
    X, y = train_df[cfg["text_column"]].fillna(""), train_df[TARGET].values
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])

    rows = []
    keys = list(grid)
    for values in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        params_fit = {**params, "ngram_range": tuple(params["ngram_range"])}
        scores = cross_validate(build_logreg_pipeline(cfg, **params_fit), X, y, cv=cv, n_jobs=n_jobs,
                                scoring={"pr_auc": "average_precision", "f1": "f1", "recall": "recall",
                                         "precision": "precision"})
        row = {"ngram_range": str(tuple(params["ngram_range"])), "min_df": params["min_df"],
               "max_features": params["max_features"] or "all"}
        for m in ("pr_auc", "f1", "recall", "precision"):
            row[f"{m}_mean"] = round(scores[f"test_{m}"].mean(), 4)
            row[f"{m}_std"] = round(scores[f"test_{m}"].std(), 4)
        rows.append(row)
        if verbose:
            print(f"{row['ngram_range']:8s} min_df={row['min_df']:<2} max_features={str(row['max_features']):6s} "
                  f"PR-AUC {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f} | F1 {row['f1_mean']:.3f}")

    table = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    if save:
        out = project_path("results/experiments/tfidf_tuning.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
    return table


if __name__ == "__main__":
    train_df, _, _ = load_splits()
    print(tune_tfidf(train_df).head(5).to_string(index=False))
