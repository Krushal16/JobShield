import warnings

from src.features.text_cleaning import clean_for_bert, clean_for_tfidf, preprocess
from helpers import make_df

warnings.filterwarnings("ignore")


def test_emscad_tokens_replaced():
    t = "Apply at #URL_86fd830a95a6# or #EMAIL_0b13a2cf# or #PHONE_ab12#"
    out = clean_for_bert(t)
    assert "urltoken" in out and "emailtoken" in out and "phonetoken" in out
    assert "#" not in out


def test_html_and_entities_removed():
    out = clean_for_bert("<b>Great</b> pay &amp; benefits")
    assert out == "Great pay & benefits"


def test_joined_words_split_but_brands_kept():
    out = clean_for_bert("skillsExperience contact us on WhatsApp or LinkedIn")
    assert "skills Experience" in out
    assert "whatsapp" in out and "Whats App" not in out
    assert "linkedin" in out


def test_tfidf_text_lowercase_no_stopwords_keeps_no():
    out = clean_for_tfidf("The job has NO fee and the pay is great", lemmatize=False)
    assert out == out.lower()
    assert "the" not in out.split() and "and" not in out.split()
    assert "no" in out.split()            # "no" is kept on purpose (e.g. "no experience")


def test_preprocess_adds_flags_and_removes_duplicates():
    df = make_df(100)
    df = __import__("pandas").concat([df, df.iloc[:5].assign(job_id=lambda d: d.job_id + 1000)])
    out = preprocess(df, lemmatize=False, verbose=False)
    assert out.attrs["preprocess_stats"]["duplicates_removed"] == 5
    assert len(out) == 100
    assert "missing_company_profile" in out.columns
    assert out["employment_type"].isna().sum() == 0      # filled with "Missing"
    assert {"text_clean", "text_bert"} <= set(out.columns)
