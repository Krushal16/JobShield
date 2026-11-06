import warnings

import numpy as np

from src.features.text_cleaning import preprocess
from src.data.split import make_splits
from src.models.baseline import build_logreg_pipeline, build_nb_pipeline, build_tfidf, fit_pipeline
from src.training.train_baseline import train_and_evaluate, best_f1_threshold, save_model, top_features
from src.training.tune_tfidf import tune_tfidf
from src.inference.predict import load_model, predict_posting
from helpers import make_df

warnings.filterwarnings("ignore")


def _splits():
    clean = preprocess(make_df(600, fake_share=0.1), lemmatize=False, verbose=False)
    return make_splits(clean, seed=42)


def test_pipelines_fit_and_predict_probabilities():
    tr, va, _ = _splits()
    for builder in (build_logreg_pipeline, build_nb_pipeline):
        pipe = fit_pipeline(builder(), tr["text_clean"], tr["fraudulent"].values)
        prob = pipe.predict_proba(va["text_clean"])[:, 1]
        assert prob.shape == (len(va),)
        assert ((prob >= 0) & (prob <= 1)).all()


def test_train_and_evaluate_returns_both_models():
    tr, va, _ = _splits()
    out = train_and_evaluate(tr, va, log=False, verbose=False)
    assert len(out["results"]) == 4                      # 2 models x (default + tuned threshold)
    assert set(out["models"]) == {"TF-IDF + Logistic Regression", "TF-IDF + Naive Bayes"}


def test_best_threshold_is_valid():
    thr, f1 = best_f1_threshold(np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.2, 0.6, 0.7, 0.9]))
    assert 0 < thr <= 1 and f1 == 1.0


def test_tfidf_is_fitted_on_training_data_only():
    tr, va, _ = _splits()
    va = va.copy()
    va["text_clean"] = va["text_clean"] + " zzqvalidationonly"
    pipe = fit_pipeline(build_logreg_pipeline(min_df=1), tr["text_clean"], tr["fraudulent"].values)
    pipe.predict_proba(va["text_clean"])
    vocab = set(pipe.named_steps["tfidf"].vocabulary_)
    assert "zzqvalidationonly" not in vocab
    assert vocab == set(build_tfidf(min_df=1).fit(tr["text_clean"]).vocabulary_)


def test_save_load_and_predict_posting(tmp_path):
    tr, _, _ = _splits()
    pipe = fit_pipeline(build_logreg_pipeline(), tr["text_clean"], tr["fraudulent"].values)
    bundle = load_model(save_model(pipe, 0.4, path=tmp_path / "m.joblib"))
    r = predict_posting(title="Job title 1", description="Posting number one, contact us on WhatsApp. Training fee required.",
                        has_company_logo=0, model_bundle=bundle)
    assert 0 <= r["probability_fake"] <= 1
    assert r["label"] in {"fraudulent", "legitimate"} and r["risk_level"] in {"Low", "Medium", "High"}
    assert r["model"] == "baseline"
    assert "No company profile" in r["red_flags"] and "No company logo" in r["red_flags"]
    assert any(f.startswith("Asks to contact by messaging app") for f in r["red_flags"])
    assert len(top_features(pipe, n=5)) == 10


def test_tune_tfidf_small_grid():
    tr, _, _ = _splits()
    grid = {"ngram_range": [[1, 1], [1, 2]], "min_df": [1], "max_features": [None]}
    table = tune_tfidf(tr, grid=grid, n_jobs=1, verbose=False, save=False)
    assert len(table) == 2 and table["pr_auc_mean"].between(0, 1).all()
