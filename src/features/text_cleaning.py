"""Week 2 text preprocessing pipeline for JobShield (src/features/text_cleaning.py).

Produces two text versions of every posting:
  * text_clean – lower-cased, cleaned, lemmatised, stop words removed  -> TF-IDF models (Weeks 3–5)
  * text_bert  – lightly cleaned, original case and punctuation kept    -> DistilBERT (Week 6)

Missing-value strategy (agreed by the team in Week 2):
  * Text fields: missing -> empty string, plus a 0/1 "missing" flag (missing text is itself a warning sign)
  * Categorical fields: missing -> the category "Missing" (kept, not dropped)
  * No columns are dropped at this stage; job_id is kept only as an identifier

Usage:
    from src.features.text_cleaning import preprocess
    clean = preprocess(df)                    # df from src.data_loader.load_emscad()
"""
import html
import re
import warnings
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.data.loader import TEXT_COLS, CATEGORICAL_COLS, BINARY_COLS, TARGET

# EMSCAD replaced real links / e-mails / phone numbers with tokens like  #URL_86fd...#
EMSCAD_TOKEN = re.compile(r"#(URL|EMAIL|PHONE)_[0-9a-fA-F]+#")
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
CAMEL_JOIN = re.compile(r"([a-z])([A-Z])")      # "skillsExperience" -> "skills Experience"
# brand names written in CamelCase must not be split ("WhatsApp" is a key scam signal)
BRANDS = re.compile(r"\b(whats\s?app|linked\s?in|pay\s?pal|you\s?tube|face\s?book|i\s?phone|java\s?script|"
                    r"git\s?hub|cash\s?app|we\s?chat|power\s?point|share\s?point|sales\s?force|word\s?press)\b", re.I)
NON_ALPHA = re.compile(r"[^a-z\s]")
SPACES = re.compile(r"\s+")

# keep a few words that matter for scam detection even though they are common
KEEP = {"no", "not", "free", "call", "online", "home", "per"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP

MISSING_FLAG_COLS = ["company_profile", "description", "requirements", "benefits",
                     "salary_range", "department", "required_education"]


@lru_cache(maxsize=1)
def _lemmatizer():
    """Return an NLTK WordNet lemmatiser, or None if NLTK is not available."""
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        for pkg in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        lem = WordNetLemmatizer()
        lem.lemmatize("jobs")  # forces WordNet to load now
        return lem
    except Exception as exc:  # pragma: no cover - depends on environment
        warnings.warn(f"Lemmatisation skipped ({exc}). Run: pip install nltk")
        return None


def _base_clean(text):
    """Steps shared by both text versions."""
    text = html.unescape(str(text))
    text = HTML_TAG.sub(" ", text)
    text = EMSCAD_TOKEN.sub(lambda m: f" {m.group(1).lower()}token ", text)
    text = URL.sub(" urltoken ", text)
    text = EMAIL.sub(" emailtoken ", text)
    text = PHONE.sub(" phonetoken ", text)
    text = BRANDS.sub(lambda m: re.sub(r"\s", "", m.group(0)).lower(), text)
    text = CAMEL_JOIN.sub(r"\1 \2", text)
    return SPACES.sub(" ", text).strip()


def clean_for_bert(text):
    """Light cleaning: keeps case and punctuation, which transformers use."""
    return _base_clean(text)


def clean_for_tfidf(text, lemmatize=True, cache={}):
    """Full cleaning for bag-of-words / TF-IDF models."""
    text = NON_ALPHA.sub(" ", _base_clean(text).lower())
    tokens = [t for t in text.split() if len(t) > 1 and t not in STOP_WORDS]
    lem = _lemmatizer() if lemmatize else None
    if lem is not None:
        out = []
        for t in tokens:
            if t not in cache:
                cache[t] = lem.lemmatize(t)
            out.append(cache[t])
        tokens = out
    return " ".join(tokens)


def preprocess(df, lemmatize=True, drop_duplicates=True, verbose=True):
    """Return a cleaned copy of the EMSCAD dataframe ready for splitting and modelling."""
    out = df.copy()
    n0 = len(out)

    # 1. missing-value flags (before filling)
    for c in MISSING_FLAG_COLS:
        out[f"missing_{c}"] = out[c].isna().astype(int)

    # 2. fill missing values
    for c in TEXT_COLS:
        out[c] = out[c].fillna("")
    for c in CATEGORICAL_COLS:
        out[c] = out[c].fillna("Missing").astype(str).str.strip().replace("", "Missing")
    for c in BINARY_COLS + [TARGET]:
        out[c] = out[c].fillna(0).astype(int)

    # 3. country from "US, NY, New York"
    out["country"] = out["location"].str.split(",").str[0].str.strip().replace("", "Missing")

    # 4. merged raw text -> two cleaned versions
    raw = out[TEXT_COLS].agg(" ".join, axis=1)
    out["text_bert"] = raw.map(clean_for_bert)
    out["text_clean"] = raw.map(lambda t: clean_for_tfidf(t, lemmatize=lemmatize))
    out["n_words_clean"] = out["text_clean"].str.split().str.len()

    # 5. remove exact duplicate postings (same text and label) so they cannot leak across splits
    n_dup = 0
    if drop_duplicates:
        dup = out.duplicated(subset=["text_bert", TARGET], keep="first")
        n_dup = int(dup.sum())
        out = out[~dup].reset_index(drop=True)

    # 6. postings with no usable text
    n_empty = int((out["n_words_clean"] == 0).sum())

    if verbose:
        print(f"Rows in: {n0:,} | exact duplicates removed: {n_dup:,} | rows out: {len(out):,}")
        print(f"Postings with no usable text after cleaning: {n_empty:,}")
        print(f"Fake postings: {out[TARGET].sum():,} ({out[TARGET].mean() * 100:.2f}%)")
    out.attrs["preprocess_stats"] = {"rows_in": n0, "duplicates_removed": n_dup,
                                     "rows_out": len(out), "empty_text": n_empty}
    return out
