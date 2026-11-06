"""Score a single job posting with a saved model (used by the Streamlit app).

    from src.inference.predict import predict_posting
    predict_posting(title="Data Entry Clerk", description="Earn $500 a week from home ...", model="xgb")
"""
import joblib
import numpy as np
import pandas as pd

from src.data.loader import EXPECTED_COLS
from src.features.red_flags import found_red_flags
from src.features.text_cleaning import preprocess
from src.utils.config import load_config, project_path

_CACHE = {}


def load_model(path=None, model="baseline"):
    """Load (and cache) a model bundle: {'model': pipeline, 'threshold': float, 'kind': ..., 'metadata': dict}."""
    path = project_path(path or load_config()["model_files"][model])
    if path not in _CACHE:
        if not path.exists():
            script = "train_xgb" if model == "xgb" else "train_baseline"
            raise FileNotFoundError(f"{path} not found – run `python -m src.training.{script}` first.")
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def risk_level(prob, threshold):
    """Consistent with the label: High = flagged as fraudulent; Medium = below the threshold but close to it."""
    if prob >= threshold:
        return "High"
    if prob >= 0.6 * threshold:
        return "Medium"
    return "Low"


def posting_frame(title="", company_profile="", description="", requirements="", benefits="",
                  has_company_logo=1, telecommuting=0, has_questions=0, salary_range=""):
    """Build a one-row processed DataFrame exactly like the training data."""
    row = {c: None for c in EXPECTED_COLS}
    row.update(job_id=0, title=title, company_profile=company_profile or None, description=description,
               requirements=requirements or None, benefits=benefits or None, salary_range=salary_range or None,
               has_company_logo=int(has_company_logo), telecommuting=int(telecommuting),
               has_questions=int(has_questions), fraudulent=0)
    lemmatize = load_config()["preprocessing"]["lemmatize"]
    return preprocess(pd.DataFrame([row]), lemmatize=lemmatize, drop_duplicates=False, verbose=False)


def _word_contributions(pipe, cleaned_text, top_n):
    """For Logistic Regression: which words in THIS posting pushed the score towards fake / real."""
    clf = pipe.named_steps.get("clf")
    if "tfidf" not in pipe.named_steps or not hasattr(clf, "coef_"):
        return [], []
    vec = pipe.named_steps["tfidf"]
    x = vec.transform([cleaned_text])
    idx = x.indices
    if len(idx) == 0:
        return [], []
    contrib = x.data * clf.coef_[0][idx]
    words = vec.get_feature_names_out()[idx]
    order = np.argsort(contrib)
    fake = [(words[i], round(float(contrib[i]), 3)) for i in order[::-1][:top_n] if contrib[i] > 0]
    real = [(words[i], round(float(contrib[i]), 3)) for i in order[:top_n] if contrib[i] < 0]
    return fake, real


def predict_posting(title="", company_profile="", description="", requirements="", benefits="",
                    has_company_logo=1, telecommuting=0, has_questions=0, salary_range="",
                    model_bundle=None, model="baseline", top_n=5):
    """Return the fake probability, label, risk level, red flags found and (for LR) the words behind it."""
    bundle = model_bundle or load_model(model=model)
    pipe, threshold, kind = bundle["model"], bundle["threshold"], bundle.get("kind", "baseline")
    df = posting_frame(title, company_profile, description, requirements, benefits,
                       has_company_logo, telecommuting, has_questions, salary_range)
    text_col = load_config()["text_column"]
    X = df if kind == "xgb" else df[text_col]
    prob = float(pipe.predict_proba(X)[0, 1])
    fake_words, real_words = _word_contributions(pipe, df[text_col].iloc[0], top_n)
    return {
        "probability_fake": round(prob, 4),
        "label": "fraudulent" if prob >= threshold else "legitimate",
        "risk_level": risk_level(prob, threshold),
        "threshold": round(threshold, 4),
        "model": kind,
        "red_flags": found_red_flags(df.iloc[0]),
        "words_towards_fake": fake_words,
        "words_towards_real": real_words,
    }
