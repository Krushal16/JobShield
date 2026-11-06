import pandas as pd

from src.features.red_flags import (RED_FLAG_COLUMNS, RedFlagTransformer, compute_red_flags, found_red_flags,
                                    matched_keywords, red_flag_fraud_rates)
from src.features.text_cleaning import preprocess
from helpers import make_df


def _one(**kw):
    base = dict(title="", company_profile="", description="", requirements="", benefits="",
                has_company_logo=1, telecommuting=0, has_questions=1)
    base.update(kw)
    return pd.DataFrame([base])


def test_columns_and_shape():
    f = compute_red_flags(_one(description="hello"))
    assert list(f.columns) == RED_FLAG_COLUMNS and len(f) == 1


def test_keyword_groups_count_whole_words_case_insensitive():
    f = compute_red_flags(_one(description="Pay the TRAINING FEE by cheque. Text us on WhatsApp. Learn more!"))
    assert f.loc[0, "payment_words"] == 2          # "training fee" and "cheque" ("fee" inside the phrase counts once)
    assert f.loc[0, "messaging_app"] == 2          # "text us" and "whatsapp"
    assert f.loc[0, "easy_money_words"] == 0       # "Learn" must not match "earn"
    assert f.loc[0, "exclamation_count"] == 1


def test_missing_fields_logo_and_title():
    f = compute_red_flags(_one(title="Data Entry Clerk", has_company_logo=0, description="x"))
    assert f.loc[0, "missing_company_profile"] == 1 and f.loc[0, "missing_salary_range"] == 1
    assert f.loc[0, "no_company_logo"] == 1 and f.loc[0, "scam_title"] == 1


def test_contact_tokens_counted_from_cleaned_text():
    clean = preprocess(make_df(5).assign(description="Email #EMAIL_abc12# or call #PHONE_ff01#"),
                       lemmatize=False, drop_duplicates=False, verbose=False)
    f = compute_red_flags(clean)
    assert (f["n_email_tokens"] == 1).all() and (f["n_phone_tokens"] == 1).all()


def test_found_red_flags_is_readable():
    flags = found_red_flags(_one(description="Earn weekly pay. Send your bank account details. Bitcoin accepted.").iloc[0])
    text = " | ".join(flags)
    assert "No company profile" in text and "Asks for personal or bank details" in text
    assert "Mentions cryptocurrency: bitcoin" in text


def test_matched_keywords():
    m = matched_keywords("Contact us on Telegram, start today")
    assert m["messaging_app"] == ["telegram"] and m["urgency_words"] == ["start today"]


def test_fraud_rate_table_and_transformer():
    clean = preprocess(make_df(300, fake_share=0.1), lemmatize=False, verbose=False)
    t = red_flag_fraud_rates(clean, min_group_size=5)
    row = t.set_index("flag").loc["missing_company_profile"]
    assert row["fraud_rate_with_%"] > row["fraud_rate_without_%"]   # fake postings have no profile in helpers
    X = RedFlagTransformer().fit_transform(clean)
    assert X.shape == (len(clean), len(RED_FLAG_COLUMNS))
    assert list(RedFlagTransformer().get_feature_names_out()) == RED_FLAG_COLUMNS
