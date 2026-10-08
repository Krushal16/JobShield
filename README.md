# JobShield – Explainable Detection of Fraudulent Job Postings

**AML-3403 AI and ML Capstone · Group 5 · Fall 2026 · Lambton College**
Faculty Supervisor: William Pourmajidi

| Member | Student ID | Core responsibility |
|---|---|---|
| Krushal Sardhara | c0966419 | Coordination, repository, live-data pipeline, Streamlit app and deployment |
| Arti Makwana | c0959575 | Data acquisition, EDA, preprocessing, red-flag features, 2026 test set quality control |
| Diksha Sharma | c0965344 | Baseline and XGBoost models, imbalance handling, evaluation, SHAP |
| Narinder Kaur | c0965335 | DistilBERT fine-tuning, LIME, drift analysis, final report integration |

Team lead rotates weekly (Week 1 Krushal → Arti → Diksha → Narinder → …).

---

## 1. Problem and objective

Fake job postings steal personal information and money from job seekers – students, newcomers and remote-work
seekers are frequent targets. **JobShield** reads a job posting and predicts whether it is **real or fraudulent**,
gives a risk score, and highlights the words and warning signs behind the decision.

## 2. Research questions

1. How well do classical models (TF-IDF + Logistic Regression, XGBoost) and a transformer (DistilBERT) detect fraudulent postings in the EMSCAD dataset?
2. Do engineered red-flag features (missing company profile or logo, contact tokens, scam phrases) improve detection over text alone?
3. How much does performance drop on recent (2025–2026) postings, and which new scam patterns are missed? (*dataset drift*)
4. Can SHAP and LIME explanations make predictions understandable to non-technical users?

## 3. Dataset

| Item | Details |
|---|---|
| Name | Employment Scam Aegean Dataset (EMSCAD), Kaggle version "Real / Fake Job Posting Prediction" |
| Source | https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction · original paper: Vidros et al. (2017), *Future Internet*, 9(1), 6 |
| Licence / usage | *To record when downloading: copy the licence shown on the Kaggle page here.* Used for academic research only. |
| Date retrieved | *To record (YYYY-MM-DD).* |
| Size | 17,880 postings, 18 columns, 866 fraudulent (4.8%); single CSV, about 50 MB |
| Collected | 2012–2014 job ads; real e-mails, phone numbers and links already replaced by codes (e.g. `#URL_…#`) |
| Storage | **Not committed to GitHub.** Download it and save as `data/raw/fake_job_postings.csv` (never edit this file) |

**2026 test set:** about 200 recent postings collected and hand-labelled by the team using
[`docs/labelling_guideline.md`](docs/labelling_guideline.md). Used **only for testing** dataset drift, never for training.

### Preprocessing (`src/features/text_cleaning.py`)
- Missing text → empty string **plus** a `missing_*` flag; missing categories → `"Missing"` (missing information is a warning sign, so it is kept)
- HTML tags and entities removed; EMSCAD link / e-mail / phone codes → `urltoken`, `emailtoken`, `phonetoken`
- Joined words split (`skillsExperience` → `skills Experience`) while brand names such as WhatsApp are protected
- Two text versions: `text_clean` (lower-case, stop words removed, lemmatised) for TF-IDF models; `text_bert` (case and punctuation kept) for DistilBERT
- Exact duplicate postings removed **before** splitting

### Split strategy (`src/data/split.py`)
Stratified **70 / 15 / 15** train / validation / test split, random seed **42** (both in `configs/config.yaml`).
Every split keeps the same fake share. The test set is used only for final model comparison.
Anything that is *learned* from data (TF-IDF vocabulary, scalers, SMOTE) is fitted **on the training set only** to avoid leakage.

## 4. Methods

| Stage | Method | Week |
|---|---|---|
| Baseline | TF-IDF (1–2-grams) + Logistic Regression (balanced class weights) | 3 |
| Classical | TF-IDF + red-flag features + XGBoost; class weights / SMOTE; 5-fold CV tuning | 4–5 |
| Transformer | Fine-tuned DistilBERT (Hugging Face) on `text_bert` | 6 |
| Explainability | SHAP (XGBoost), LIME (DistilBERT) | 8 |
| Application | Streamlit app (Check a Posting, Live Feed, Model Insights) on Hugging Face Spaces | 9–10 |

**Evaluation metrics** (`src/evaluation/metrics.py`): precision, recall, F1 and PR-AUC for the fraudulent class
(main), ROC-AUC and accuracy for reference. Accuracy alone is misleading: predicting "real" for everything is about 95% accurate.

## 5. Repository structure

```
JobShield/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                # template for API keys (real keys go in .env, never committed)
├── .github/pull_request_template.md
├── configs/config.yaml         # paths, seed, split ratios, model parameters
├── docs/                       # labelling guideline, 2026 test set template, GitHub workflow
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_preprocessing.ipynb
├── src/
│   ├── data/                   # loader.py (load + validate), split.py (stratified split)
│   ├── features/               # text_cleaning.py (Week 2); red_flags.py (Week 4)
│   ├── models/                 # model definitions (Weeks 3–6)
│   ├── training/               # training and cross-validation scripts (Weeks 3–6)
│   ├── evaluation/             # metrics.py (shared metrics, plots, results log)
│   ├── inference/              # predict_posting() for the app (Week 3+)
│   └── utils/                  # config.py (config loading, paths, seeds)
├── data/
│   ├── raw/                    # original Kaggle CSV (not in Git)
│   ├── interim/                # cleaned data before splitting
│   ├── processed/              # train.csv, val.csv, test.csv
│   └── external/               # 2026 test set export
├── models/checkpoints/ , models/final/     # model files (not in Git – see section 8)
├── results/
│   ├── metrics/results.csv     # one row per model run
│   ├── figures/                # charts used in reports
│   ├── predictions/            # saved model predictions
│   └── experiments/            # comparison and ablation summaries
├── tests/                      # pytest tests for loading, cleaning, splitting, metrics
├── logs/                       # run logs (not in Git; findings go in the research log)
└── research_log/research_log.md
```

## 6. Installation

```bash
git clone <repo-url>
cd JobShield
pip install -r requirements.txt
```
Google Colab: `!git clone <repo-url>`, `%cd JobShield`, `!pip install -r requirements.txt`, then upload the CSV to `data/raw/`.

## 7. How to run

1. Put `fake_job_postings.csv` in `data/raw/`.
2. Run `notebooks/01_data_exploration.ipynb` → charts in `results/figures/`.
3. Run `notebooks/02_preprocessing.ipynb` → `data/interim/emscad_clean.csv` and `data/processed/{train,val,test}.csv`.
4. Run the tests: `pytest tests/`.

## 8. Models and large files
Trained models are saved to `models/final/` and are **not** committed to Git. Final models will be published on
Hugging Face Hub (link added here in Week 10).

## 9. Main results
*Updated each week from `results/metrics/results.csv`.*

| Model | Split | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Always predict "real" (dummy) | val | 0.00 | 0.00 | 0.00 | – |
| TF-IDF + Logistic Regression | val | Week 3 | | | |

## 10. Known limitations
- EMSCAD postings are from 2012–2014; scam styles have changed (measured with the 2026 test set).
- Only 866 fraudulent examples, so results on the minority class have high variance.
- Real e-mail addresses are hidden in EMSCAD, so e-mail domain red flags cannot be learned from it.
- The 2026 test set is small (about 200 postings) and labelled by the team, so it may contain labelling errors.

## 11. Team workflow
See [`docs/github_workflow.md`](docs/github_workflow.md): one branch per task, `type: description` commit messages,
pull request reviewed by another member before merging to `main`, research log updated for every significant experiment.

## References
Vidros, S., Kolias, C., Kambourakis, G., & Akoglu, L. (2017). Automatic detection of online recruitment frauds:
Characteristics, methods, and a public dataset. *Future Internet, 9*(1), 6. https://doi.org/10.3390/fi9010006
