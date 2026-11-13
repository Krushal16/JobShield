"""Week 4: compare the Week 3 baseline with XGBoost, with and without red-flag features (ablation).

    python -m src.training.train_xgb

Outputs
    results/metrics/results.csv                        appended rows for all three models
    results/experiments/red_flag_fraud_rates.csv       fraud rate with / without each red flag (training set)
    results/experiments/xgb_feature_importance.csv     top features of the best XGBoost model
    results/predictions/val_xgb_predictions.csv
    models/final/xgb_tfidf_redflags.joblib             saved model + threshold (not in Git)
"""
from src.data.loader import TARGET
from src.data.split import load_splits
from src.evaluation.metrics import evaluate, log_result, print_report
from src.features.red_flags import red_flag_fraud_rates
from src.models.baseline import build_logreg_pipeline, fit_pipeline
from src.models.xgb import build_xgb_pipeline, feature_importance, fit_xgb
from src.training.train_baseline import best_f1_threshold, save_model
from src.utils.config import load_config, project_path, set_seed

BASELINE = "TF-IDF + Logistic Regression (Week 3 baseline)"
XGB_TEXT = "TF-IDF + XGBoost"
XGB_FLAGS = "TF-IDF + red flags + XGBoost"


def train_and_compare(train_df, val_df, cfg=None, log=True, verbose=True):
    """Fit the three models on train, score on validation at threshold 0.5 and at the F1-best threshold."""
    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    col, y_tr, y_va = cfg["text_column"], train_df[TARGET].values, val_df[TARGET].values

    fitted = {
        BASELINE: fit_pipeline(build_logreg_pipeline(cfg), train_df[col].fillna(""), y_tr, cfg),
        XGB_TEXT: fit_xgb(build_xgb_pipeline(cfg, use_red_flags=False), train_df, y_tr, cfg),
        XGB_FLAGS: fit_xgb(build_xgb_pipeline(cfg, use_red_flags=True), train_df, y_tr, cfg),
    }
    out = {"results": [], "models": fitted, "probs": {}, "thresholds": {}}
    for name, pipe in fitted.items():
        X_va = val_df[col].fillna("") if name == BASELINE else val_df
        prob = pipe.predict_proba(X_va)[:, 1]
        best_thr, _ = best_f1_threshold(y_va, prob)
        for thr, note in ((cfg["threshold"]["default"], "threshold=0.5"), (best_thr, f"F1-best threshold={best_thr:.3f}")):
            r = evaluate(y_va, (prob >= thr).astype(int), y_score=prob, model_name=name, split="val", notes=note)
            out["results"].append(r)
            if log:
                log_result(r)
            if verbose:
                print_report(r)
        out["probs"][name], out["thresholds"][name] = prob, best_thr
    return out


def main():
    cfg = load_config()
    train_df, val_df, _ = load_splits()              # test set stays untouched until Week 7
    exp = project_path("results/experiments"); exp.mkdir(parents=True, exist_ok=True)
    red_flag_fraud_rates(train_df).to_csv(exp / "red_flag_fraud_rates.csv", index=False)

    out = train_and_compare(train_df, val_df, cfg)
    pipe, thr = out["models"][XGB_FLAGS], out["thresholds"][XGB_FLAGS]
    print("Saved:", save_model(pipe, thr, cfg=cfg, kind="xgb", extra={"xgboost": cfg["xgboost"]}))
    feature_importance(pipe).to_csv(exp / "xgb_feature_importance.csv", index=False)

    pred_dir = project_path(cfg["paths"]["predictions_dir"]); pred_dir.mkdir(parents=True, exist_ok=True)
    preds = val_df[["job_id", "title", TARGET]].copy()
    for key, name in (("baseline", BASELINE), ("xgb_text", XGB_TEXT), ("xgb_flags", XGB_FLAGS)):
        preds[f"prob_{key}"] = out["probs"][name]
    preds.to_csv(pred_dir / "val_xgb_predictions.csv", index=False)


if __name__ == "__main__":
    main()
