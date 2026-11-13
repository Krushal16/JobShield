"""Week 3: train and evaluate the baseline models on the fixed train/validation split.

Run from the project root:
    python -m src.training.train_baseline

Outputs
    results/metrics/results.csv                     one row per model (appended)
    results/predictions/val_baseline_predictions.csv
    results/experiments/baseline_top_features.csv
    models/final/baseline_tfidf_logreg.joblib       best Logistic Regression model + threshold (not in Git)
"""
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.data.loader import TARGET
from src.data.split import load_splits
from src.evaluation.metrics import evaluate, log_result, print_report
from src.models.baseline import build_logreg_pipeline, build_nb_pipeline, fit_pipeline
from src.utils.config import load_config, project_path, set_seed

MODELS = {
    "TF-IDF + Logistic Regression": build_logreg_pipeline,
    "TF-IDF + Naive Bayes": build_nb_pipeline,
}


def best_f1_threshold(y_true, y_prob):
    """Threshold on predicted probability that gives the highest F1 (chosen on validation data only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return (float(thresholds[i]) if len(thresholds) else 0.5), float(f1[i])


def train_and_evaluate(train_df, val_df, cfg=None, log=True, verbose=True):
    """Fit every baseline on train, score on validation. Returns results, fitted models and probabilities."""
    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    col = cfg["text_column"]
    X_tr, y_tr = train_df[col].fillna(""), train_df[TARGET].values
    X_va, y_va = val_df[col].fillna(""), val_df[TARGET].values

    out = {"results": [], "models": {}, "probs": {}, "thresholds": {}}
    for name, builder in MODELS.items():
        pipe = fit_pipeline(builder(cfg), X_tr, y_tr, cfg)
        prob = pipe.predict_proba(X_va)[:, 1]
        thr = cfg["threshold"]["default"]
        res = evaluate(y_va, (prob >= thr).astype(int), y_score=prob, model_name=name, split="val",
                       notes=f"threshold={thr}")
        best_thr, best_f1 = best_f1_threshold(y_va, prob)
        res_tuned = evaluate(y_va, (prob >= best_thr).astype(int), y_score=prob, model_name=name, split="val",
                             notes=f"F1-best threshold={best_thr:.3f}")
        for r in (res, res_tuned):
            out["results"].append(r)
            if log:
                log_result(r)
            if verbose:
                print_report(r)
        out["models"][name], out["probs"][name], out["thresholds"][name] = pipe, prob, best_thr
    return out


def save_model(pipe, threshold, path=None, cfg=None, extra=None):
    """Save the pipeline together with its decision threshold and basic metadata."""
    cfg = cfg or load_config()
    path = project_path(path or cfg["model_files"]["baseline"])
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": pipe, "threshold": float(threshold),
              "metadata": {"trained": datetime.now().isoformat(timespec="minutes"),
                           "text_column": cfg["text_column"], "tfidf": cfg["tfidf"],
                           "logistic_regression": cfg["logistic_regression"], "seed": cfg["seed"], **(extra or {})}}
    joblib.dump(bundle, path)
    return path


def top_features(pipe, n=25):
    """Words with the largest Logistic Regression weights towards 'fake' and towards 'real'."""
    words = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    fake = pd.DataFrame({"word": words[order[::-1][:n]], "weight": coef[order[::-1][:n]], "direction": "fake"})
    real = pd.DataFrame({"word": words[order[:n]], "weight": coef[order[:n]], "direction": "real"})
    return pd.concat([fake, real], ignore_index=True)


def main():
    cfg = load_config()
    train_df, val_df, _ = load_splits()          # the test set is NOT touched until Week 7
    out = train_and_evaluate(train_df, val_df, cfg)

    name = "TF-IDF + Logistic Regression"
    pipe, thr = out["models"][name], out["thresholds"][name]
    path = save_model(pipe, thr, cfg=cfg)
    print(f"Saved {name} to {path}")

    pred_dir = project_path(cfg["paths"]["predictions_dir"]); pred_dir.mkdir(parents=True, exist_ok=True)
    preds = val_df[["job_id", "title", TARGET]].copy()
    for n, p in out["probs"].items():
        preds[f"prob_{'logreg' if 'Logistic' in n else 'nb'}"] = p
    preds.to_csv(pred_dir / "val_baseline_predictions.csv", index=False)

    exp_dir = project_path("results/experiments"); exp_dir.mkdir(parents=True, exist_ok=True)
    top_features(pipe).to_csv(exp_dir / "baseline_top_features.csv", index=False)
    print("Saved validation predictions and top features.")


if __name__ == "__main__":
    main()
