# JobShield – Research Log

One entry per significant experiment or decision, including ones that did not work.
Newest entries at the bottom. Fill in "Related Git commit or Pull Request" with the PR number after merging.

---

## Entry 01 – Week 1 – Dataset validation and first exploration

**Date:** 2026-10-02
**Objective / research question:** Understand the EMSCAD dataset before modelling: size, class balance, missing data and early warning signs.
**Hypothesis:** Fraudulent postings leave company information empty more often and use different vocabulary from real postings.
**Dataset and version:** EMSCAD, Kaggle "Real / Fake Job Posting Prediction", `fake_job_postings.csv` (retrieved: *add date*).
**Model / method:** Exploratory data analysis – counts, missing-value rates by class, fraud rate by flag, text length, word frequency.
**Parameters:** Categories with fewer than 50 postings excluded from fraud-rate tables.
**Changes made:** Created `src/data/loader.py` (schema validation) and `notebooks/01_data_exploration.ipynb`.
**Results:** 17,880 postings, 18 columns, 866 fraudulent (4.8%). *Add from notebook: top missing fields, fraud rate with vs without company logo, words most over-represented in fake postings.*
**Interpretation:** The data is highly imbalanced, so accuracy is not a useful metric; precision, recall, F1 and PR-AUC for the fraudulent class will be the main metrics.
**Problems encountered:** Dataset too large to commit; kept out of Git and shared through one agreed Google Drive copy.
**Next step:** Finish EDA on categorical fields; decide the missing-value strategy; build the preprocessing pipeline.
**Related Git commit or Pull Request:** *#…*

---

## Entry 02 – Week 2 – Missing-value strategy and text preprocessing

**Date:** 2026-10-09
**Objective / research question:** Prepare clean, leakage-free inputs for both TF-IDF models and DistilBERT.
**Hypothesis:** Missing fields carry information about fraud, so dropping them would remove useful signal.
**Dataset and version:** EMSCAD raw CSV → `data/interim/emscad_clean.csv`.
**Model / method:** Rule-based cleaning in `src/features/text_cleaning.py`.
**Parameters:** `lemmatize: true`, `drop_duplicates: true` (configs/config.yaml); sklearn English stop words minus {no, not, free, call, online, home, per}.
**Changes made:**
- Missing text → empty string + `missing_*` flag; missing categories → "Missing"; no rows dropped for missing values.
- EMSCAD codes (`#URL_…#`, `#EMAIL_…#`, `#PHONE_…#`) → `urltoken` / `emailtoken` / `phonetoken`.
- Joined words split (`skillsExperience` → `skills Experience`).
- Two outputs: `text_clean` (TF-IDF) and `text_bert` (DistilBERT).
- Exact duplicates (same text and label) removed before splitting.
**Results:** *Add from notebook: number of duplicates removed, rows after cleaning, share of fake postings with missing company profile vs real.*
**Interpretation:** Keeping missing-value flags lets later models use "missing company profile" as a warning sign.
**Problems encountered:** The joined-word rule split brand names – "WhatsApp" became "Whats App", which would hide a key scam signal. Fixed with a protected brand-name list and a unit test (`tests/test_text_cleaning.py`). NLTK WordNet must be downloaded in each Colab session – added an automatic download.
**Next step:** Stratified split and evaluation functions.
**Related Git commit or Pull Request:** *#…*

---

## Entry 03 – Week 2 – Data split and dummy baselines

**Date:** 2026-10-10
**Objective / research question:** Create one fixed split for all models and show why accuracy is misleading.
**Hypothesis:** A model that always predicts "real" will score about 95% accuracy but catch no scams.
**Dataset and version:** `data/interim/emscad_clean.csv` → `data/processed/{train,val,test}.csv`.
**Model / method:** Stratified 70/15/15 split (`src/data/split.py`); dummy baselines scored with `src/evaluation/metrics.py`.
**Parameters:** seed 42; split 0.70 / 0.15 / 0.15.
**Changes made:** Split checks (no posting in two splits, equal fake share); shared results log `results/metrics/results.csv`.
**Results:** "Always predict real" on validation: accuracy ≈ 95%, recall 0, F1 0. *Add exact split sizes from notebook.*
**Interpretation:** Confirms F1 and PR-AUC for the fraudulent class as main metrics. With about 130 fake postings in validation, scores will vary, so 5-fold cross-validation on the training set will be used for tuning.
**Problems encountered:** None.
**Next step:** Week 3 – TF-IDF + Logistic Regression baseline (TF-IDF fitted on the training set only).
**Related Git commit or Pull Request:** *#…*

---

<!-- Template for new entries
## Entry NN – Week N – Title
**Date:**
**Objective / research question:**
**Hypothesis:**
**Dataset and version:**
**Model / method:**
**Parameters:**
**Changes made:**
**Results:**
**Interpretation:**
**Problems encountered:**
**Next step:**
**Related Git commit or Pull Request:**
-->
