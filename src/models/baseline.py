"""Week 3 baseline models: TF-IDF + Logistic Regression and TF-IDF + Multinomial Naive Bayes.

Both are scikit-learn Pipelines, so the TF-IDF vocabulary is learned only from the data passed to
.fit() – the training set – which prevents leakage into validation and test.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.utils.config import load_config


def build_tfidf(cfg=None, **overrides):
    """TF-IDF vectoriser from configs/config.yaml; keyword arguments override config values."""
    cfg = cfg or load_config()
    p = {**cfg["tfidf"], **overrides}
    return TfidfVectorizer(ngram_range=tuple(p["ngram_range"]), min_df=p["min_df"], max_df=p.get("max_df", 1.0),
                           max_features=p["max_features"], sublinear_tf=p.get("sublinear_tf", True))


def build_logreg_pipeline(cfg=None, **tfidf_overrides):
    cfg = cfg or load_config()
    lr = cfg["logistic_regression"]
    return Pipeline([
        ("tfidf", build_tfidf(cfg, **tfidf_overrides)),
        ("clf", LogisticRegression(class_weight=lr["class_weight"], C=lr["C"], max_iter=lr["max_iter"],
                                   solver="liblinear", random_state=cfg["seed"])),
    ])


def build_nb_pipeline(cfg=None, **tfidf_overrides):
    cfg = cfg or load_config()
    return Pipeline([
        ("tfidf", build_tfidf(cfg, **tfidf_overrides)),
        ("clf", MultinomialNB(alpha=cfg["naive_bayes"]["alpha"])),
    ])


def fit_pipeline(pipe, X, y, cfg=None):
    """Fit a pipeline. Naive Bayes has no class_weight option, so balanced sample weights are used instead."""
    cfg = cfg or load_config()
    if isinstance(pipe.named_steps["clf"], MultinomialNB) and cfg["naive_bayes"].get("balanced_sample_weight", True):
        return pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)
