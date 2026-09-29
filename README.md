# JobShield – Explainable Detection of Fraudulent Job Postings

**AML-3403 AI and ML Capstone · Group 5 · Fall 2026**
Krushal Sardhara · Arti Makwana · Diksha Sharma · Narinder Kaur
Faculty Supervisor: William Pourmajidi

JobShield reads an online job posting and predicts whether it is **real or fraudulent**, gives a risk score,
and highlights the words and warning signs behind the decision.

## Project structure

```
JobShield/
├── data/
│   ├── raw/            # fake_job_postings.csv goes here (not committed to Git)
│   └── processed/      # cleaned data created by our notebooks
├── notebooks/
│   └── 01_eda.ipynb    # Week 1: exploratory data analysis
├── src/
│   └── data_loader.py  # load + validate the dataset, shared helper functions
├── app/                # Streamlit web app (Weeks 9–10)
├── docs/
│   └── labelling_guideline.md   # rules for labelling the 2026 test set
├── reports/figures/    # charts saved by the notebooks (used in weekly reports)
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository: `git clone <repo-url>` and `cd JobShield`
2. Install packages: `pip install -r requirements.txt`
3. Download `fake_job_postings.csv` from
   [Kaggle – Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
   and save it as `data/raw/fake_job_postings.csv`
4. Open `notebooks/01_eda.ipynb` and run all cells. Charts are saved to `reports/figures/`.

**Google Colab:** upload the repository (or `git clone` inside Colab), upload the CSV to `data/raw/`, then run the notebook.

## Dataset

Employment Scam Aegean Dataset (EMSCAD): 17,880 job postings, 866 fraudulent (4.8%), 18 columns
(Vidros et al., 2017).

## Team workflow

- One branch per task: `git checkout -b week1-eda-arti`
- Open a Pull Request into `main`; another member reviews before merging
- Every task is a GitHub issue on the project board (To do → In progress → Done)
- The weekly lead updates the board and submits the weekly report

## Weekly plan

| Week | Dates (2026) | Lead | Focus |
|---|---|---|---|
| 1 | Sep 28 – Oct 4 | Krushal | Repository, dataset, initial EDA |
| 2 | Oct 5 – 11 | Arti | Complete EDA, preprocessing, labelling guideline |
| 3 | Oct 12 – 18 | Diksha | TF-IDF + Logistic Regression baseline |
| 4 | Oct 19 – 25 | Narinder | Red-flag features, XGBoost |
| 5 | Oct 26 – Nov 1 | Krushal | Imbalance handling, tuning |
| 6 | Nov 2 – 8 | Arti | DistilBERT |
| 7 | Nov 9 – 15 | Diksha | Model comparison, error analysis |
| 8 | Nov 16 – 22 | Narinder | SHAP / LIME, drift evaluation |
| 9 | Nov 23 – 29 | Krushal | Streamlit app |
| 10 | Nov 30 – Dec 6 | Arti | Live feed, deployment |
| 11 | Dec 7 – 13 | Diksha | Testing, final report draft |
| 12 | Dec 14 – 20 | Narinder | Presentation, final submission |
