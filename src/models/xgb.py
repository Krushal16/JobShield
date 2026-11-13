"""Week 4: XGBoost on TF-IDF text features, optionally combined with red-flag features.

The whole thing is one scikit-learn Pipeline (ColumnTransformer -> XGBClassifier), so TF-IDF is learned
from the training set only and the same object scores new postings in the app.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.features.red_flags import RED_FLAG_INPUT_COLUMNS, RedFlagTransformer
from src.models.baseline import build_tfidf
from src.utils.config import load_config


def build_xgb_pipeline(cfg=None, use_red_flags=True, **tfidf_overrides):
    """TF-IDF (+ red flags) -> XGBoost. Input is a DataFrame of processed postings."""
    from xgboost import XGBClassifier          # imported here so the rest of the project works without xgboost

    cfg = cfg or load_config()
    parts = [("tfidf", build_tfidf(cfg, **tfidf_overrides), cfg["text_column"])]
    if use_red_flags:
        parts.append(("flags", RedFlagTransformer(), RED_FLAG_INPUT_COLUMNS))
    p = cfg["xgboost"]
    clf = XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                        min_child_weight=p["min_child_weight"], eval_metric=p["eval_metric"],
                        tree_method=p["tree_method"], random_state=cfg["seed"], n_jobs=-1)
    return Pipeline([("features", ColumnTransformer(parts, sparse_threshold=1.0)), ("clf", clf)])


def fit_xgb(pipe, X, y, cfg=None):
    """Fit with scale_pos_weight = (#real / #fake) in the training data when the config says 'auto'."""
    cfg = cfg or load_config()
    spw = cfg["xgboost"]["scale_pos_weight"]
    if spw == "auto":
        y = np.asarray(y)
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    pipe.set_params(clf__scale_pos_weight=spw)
    return pipe.fit(X, y)


def feature_importance(pipe, n=25):
    """Top features by XGBoost importance, with the source of each feature (word or red flag)."""
    names = pipe.named_steps["features"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:n]
    return pd.DataFrame({
        "feature": [names[i].split("__", 1)[-1] for i in order],
        "source": ["red flag" if names[i].startswith("flags__") else "word" for i in order],
        "importance": np.round(imp[order], 5),
    })
