import warnings

import pytest

xgboost = pytest.importorskip("xgboost")

from src.data.split import make_splits
from src.features.text_cleaning import preprocess
from src.inference.predict import load_model, predict_posting
from src.models.xgb import build_xgb_pipeline, feature_importance, fit_xgb
from src.training.train_baseline import save_model
from src.training.train_xgb import BASELINE, XGB_FLAGS, XGB_TEXT, train_and_compare
from helpers import make_df

warnings.filterwarnings("ignore")


def _splits():
    clean = preprocess(make_df(600, fake_share=0.1), lemmatize=False, verbose=False)
    return make_splits(clean, seed=42)


def test_xgb_with_and_without_red_flags():
    tr, va, _ = _splits()
    for flags in (False, True):
        pipe = fit_xgb(build_xgb_pipeline(use_red_flags=flags), tr, tr["fraudulent"].values)
        prob = pipe.predict_proba(va)[:, 1]
        assert prob.shape == (len(va),) and ((prob >= 0) & (prob <= 1)).all()
        assert pipe.named_steps["clf"].get_params()["scale_pos_weight"] > 1


def test_feature_importance_marks_red_flags():
    tr, _, _ = _splits()
    pipe = fit_xgb(build_xgb_pipeline(use_red_flags=True), tr, tr["fraudulent"].values)
    imp = feature_importance(pipe, n=200)
    assert set(imp.source) <= {"word", "red flag"} and "red flag" in set(imp.source)


def test_train_and_compare_three_models():
    tr, va, _ = _splits()
    out = train_and_compare(tr, va, log=False, verbose=False)
    assert set(out["models"]) == {BASELINE, XGB_TEXT, XGB_FLAGS}
    assert len(out["results"]) == 6


def test_predict_posting_with_xgb(tmp_path):
    tr, _, _ = _splits()
    pipe = fit_xgb(build_xgb_pipeline(use_red_flags=True), tr, tr["fraudulent"].values)
    bundle = load_model(save_model(pipe, 0.5, path=tmp_path / "x.joblib", kind="xgb"))
    r = predict_posting(title="Data Entry Assistant", description="Earn from home, message us on WhatsApp.",
                        has_company_logo=0, model_bundle=bundle)
    assert r["model"] == "xgb" and 0 <= r["probability_fake"] <= 1
    assert "Job title often used in scams" in r["red_flags"]
    assert r["words_towards_fake"] == []            # word contributions are only for Logistic Regression
