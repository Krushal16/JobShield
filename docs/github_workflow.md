# JobShield – GitHub Workflow (follows the course AI/ML GitHub guidelines)

## The flow for every task
**Issue → Branch → Develop → Commit → Push → Pull Request → Review → Merge to `main`**

1. **Issue** – every task is a GitHub Issue on the project board, with an owner and the week number.
2. **Branch** – one branch per task, created from the latest `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/text-preprocessing
   ```
3. **Develop** – keep unrelated work out of the branch. Update `research_log/research_log.md` for significant experiments.
4. **Commit** – small, meaningful commits (see message rules below).
5. **Push** – `git push -u origin feature/text-preprocessing`
6. **Pull Request** – fill in the template (what changed, why, how tested, results, limitations). Link the issue with `Closes #12`.
7. **Review** – **another member** reviews before merging. Run `pytest tests/` before approving.
8. **Merge** – squash or merge into `main`, delete the branch, move the issue to *Done*.

`main` must always run. Nobody pushes directly to `main`.

## Branch names
| Prefix | Use for | Example |
|---|---|---|
| `feature/` | new code | `feature/text-preprocessing`, `feature/red-flag-features` |
| `experiment/` | model experiments | `experiment/tfidf-logreg-baseline`, `experiment/distilbert` |
| `fix/` | bug fixes | `fix/whatsapp-split` |
| `data/` | dataset work | `data/2026-test-set` |
| `docs/` | documentation, reports | `docs/research-log-week2` |

## Commit messages: `<type>: <short description>`
| Type | Example |
|---|---|
| `feat` | `feat: add text cleaning pipeline with two text versions` |
| `data` | `data: create stratified 70/15/15 split` |
| `model` | `model: add TF-IDF logistic regression baseline` |
| `experiment` | `experiment: compare class weights and SMOTE` |
| `fix` | `fix: keep WhatsApp as one word during cleaning` |
| `test` | `test: add unit tests for split and metrics` |
| `docs` | `docs: update research log for entry 03` |
| `chore` | `chore: restructure repository to course guidelines` |

Never use vague messages such as *update*, *changes*, *final*, *final2*.

## Who commits what
Each member commits **their own work from their own GitHub account**, so the history shows everyone's contribution.
Do not upload the whole project from one account.

## Secrets
API keys (Reddit, Adzuna) go in a local `.env` file, which is in `.gitignore`. Copy `.env.example` to `.env` and fill it in.
If a key is ever committed by mistake, tell the team and **regenerate the key immediately**.

## Large files
The dataset and trained models are never committed. Data: see README section 3. Models: Hugging Face Hub (Week 10).
