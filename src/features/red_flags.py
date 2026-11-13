"""Week 4: red-flag (warning-sign) features for fraudulent job postings.

Bag-of-words models only see individual words. These features capture what is MISSING from a posting
and structural warning signs. Keyword lists live in configs/config.yaml (red_flags section).

Every feature is computed from a single posting only, so there is no leakage between splits.

    from src.features.red_flags import compute_red_flags, found_red_flags
    flags = compute_red_flags(df)          # one row of numeric features per posting
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.config import load_config

TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
MISSING_FIELDS = ["company_profile", "salary_range", "requirements", "benefits"]

# columns the transformer needs from the processed data (all created by src/features/text_cleaning.py)
RED_FLAG_INPUT_COLUMNS = TEXT_FIELDS + ["text_bert", "has_company_logo", "telecommuting", "has_questions"] + \
    [f"missing_{c}" for c in MISSING_FIELDS]

KEYWORD_GROUPS = ["payment_words", "messaging_app", "easy_money_words", "urgency_words",
                  "personal_info_request", "crypto_words"]

RED_FLAG_COLUMNS = (
    [f"missing_{c}" for c in MISSING_FIELDS]
    + ["no_company_logo", "telecommuting", "no_screening_questions",
       "n_email_tokens", "n_phone_tokens", "n_url_tokens",
       "company_profile_words", "description_words"]
    + KEYWORD_GROUPS
    + ["scam_title", "caps_ratio", "exclamation_count"]
)

# readable names for the app and reports
LABELS = {
    "missing_company_profile": "No company profile",
    "missing_salary_range": "No salary given",
    "missing_requirements": "No requirements listed",
    "missing_benefits": "No benefits listed",
    "no_company_logo": "No company logo",
    "telecommuting": "Remote / work-from-home job",
    "no_screening_questions": "No screening questions",
    "payment_words": "Mentions fees, deposits or cheques",
    "messaging_app": "Asks to contact by messaging app",
    "easy_money_words": "Easy-money / no-experience language",
    "urgency_words": "Urgent hiring language",
    "personal_info_request": "Asks for personal or bank details",
    "crypto_words": "Mentions cryptocurrency",
    "scam_title": "Job title often used in scams",
    "n_email_tokens": "Contains e-mail addresses",
    "n_phone_tokens": "Contains phone numbers",
    "n_url_tokens": "Contains links",
}


def _pattern(phrases):
    parts = [r"\s+".join(map(re.escape, p.lower().split())) for p in phrases]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


@lru_cache(maxsize=1)
def _patterns():
    cfg = load_config()["red_flags"]
    pats = {g: _pattern(cfg["keywords"][g]) for g in KEYWORD_GROUPS}
    pats["scam_title"] = _pattern(cfg["scam_titles"])
    return pats


def _col(df, name, default=""):
    return df[name] if name in df else pd.Series(default, index=df.index)


def _text(df):
    """Posting text used for keyword matching (text_bert keeps tokens like 'emailtoken'; else raw fields)."""
    if "text_bert" in df:
        return df["text_bert"].fillna("").astype(str)
    return df[[c for c in TEXT_FIELDS if c in df]].fillna("").astype(str).agg(" ".join, axis=1)


def _is_missing(df, field):
    if f"missing_{field}" in df:
        return df[f"missing_{field}"].fillna(1).astype(int)
    return _col(df, field).fillna("").astype(str).str.strip().eq("").astype(int)


def compute_red_flags(df):
    """Return a DataFrame with one numeric red-flag feature per column (see RED_FLAG_COLUMNS)."""
    pats, text = _patterns(), _text(df)
    lower = text.str.lower()
    out = pd.DataFrame(index=df.index)
    for f in MISSING_FIELDS:
        out[f"missing_{f}"] = _is_missing(df, f)
    out["no_company_logo"] = 1 - pd.to_numeric(_col(df, "has_company_logo", 1), errors="coerce").fillna(1).astype(int)
    out["telecommuting"] = pd.to_numeric(_col(df, "telecommuting", 0), errors="coerce").fillna(0).astype(int)
    out["no_screening_questions"] = 1 - pd.to_numeric(_col(df, "has_questions", 1), errors="coerce").fillna(1).astype(int)
    out["n_email_tokens"] = lower.str.count(r"\bemailtoken\b")
    out["n_phone_tokens"] = lower.str.count(r"\bphonetoken\b")
    out["n_url_tokens"] = lower.str.count(r"\burltoken\b")
    out["company_profile_words"] = _col(df, "company_profile").fillna("").astype(str).str.split().str.len()
    out["description_words"] = _col(df, "description").fillna("").astype(str).str.split().str.len()
    for g in KEYWORD_GROUPS:
        out[g] = lower.map(lambda t, p=pats[g]: len(p.findall(t)))
    out["scam_title"] = _col(df, "title").fillna("").astype(str).map(lambda t: int(bool(pats["scam_title"].search(t))))
    letters = text.str.count(r"[A-Za-z]")
    out["caps_ratio"] = (text.str.count(r"[A-Z]") / letters.replace(0, np.nan)).fillna(0).round(4)
    out["exclamation_count"] = text.str.count("!")
    return out[RED_FLAG_COLUMNS]


def matched_keywords(text):
    """Which keywords from each group appear in one piece of text (used to explain predictions)."""
    pats = _patterns()
    return {g: sorted({m.lower() for m in pats[g].findall(text or "")}) for g in KEYWORD_GROUPS}


def found_red_flags(row):
    """Human-readable list of the warning signs present in ONE posting (a dict or pandas Series)."""
    one = pd.DataFrame([dict(row)])
    flags = compute_red_flags(one).iloc[0]
    words = matched_keywords(_text(one).iloc[0])
    found = []
    for col in ["missing_company_profile", "no_company_logo", "missing_salary_range", "telecommuting"]:
        if flags[col] > 0:
            found.append(LABELS[col])
    for g in KEYWORD_GROUPS:
        if flags[g] > 0:
            found.append(f"{LABELS[g]}: {', '.join(words[g][:4])}")
    if flags["scam_title"]:
        found.append(LABELS["scam_title"])
    if flags["n_email_tokens"] > 0:
        found.append(LABELS["n_email_tokens"])
    return found


def red_flag_fraud_rates(df, target="fraudulent", min_group_size=None):
    """For each binary-style flag: fraud rate when the flag is present vs absent (EDA table)."""
    min_group_size = min_group_size or load_config()["red_flags"]["min_group_size"]
    flags = compute_red_flags(df)
    y = df[target].values
    rows = []
    skip = {"company_profile_words", "description_words", "caps_ratio"}
    for col in RED_FLAG_COLUMNS:
        if col in skip:
            continue
        present = flags[col].values > 0
        n_present = int(present.sum())
        rows.append({"flag": col, "label": LABELS.get(col, col), "postings_with_flag": n_present,
                     "share_%": round(present.mean() * 100, 2),
                     "fraud_rate_with_%": round(y[present].mean() * 100, 2) if n_present else np.nan,
                     "fraud_rate_without_%": round(y[~present].mean() * 100, 2) if (~present).any() else np.nan,
                     "enough_data": n_present >= min_group_size})
    t = pd.DataFrame(rows)
    t["lift"] = (t["fraud_rate_with_%"] / t["fraud_rate_without_%"]).replace([np.inf, -np.inf], np.nan).round(2)
    return t.sort_values("lift", ascending=False).reset_index(drop=True)


class RedFlagTransformer(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: DataFrame of postings -> sparse matrix of red-flag features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.csr_matrix(compute_red_flags(pd.DataFrame(X)).values.astype(np.float32))

    def get_feature_names_out(self, input_features=None):
        return np.array(RED_FLAG_COLUMNS, dtype=object)
