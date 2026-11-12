# Team Workflow

## Who uploads what
| Member | Role | Files |
|---|---|---|
| 1 | Cleaning & EDA | `notebooks/01_cleaning_eda.ipynb`, `report/01_dataset_cleaning_eda.md` |
| 2 | External data | `notebooks/02_external_data.ipynb`, `scripts/load_external_data.py`, `report/02_external_data.md` |
| 3 | Scraping | `notebooks/03_scraping_validation.ipynb`, `scripts/scrape_scores.py`, `report/03_scraping_ethics.md` |
| 4 | Features & merging | `notebooks/04_feature_engineering.ipynb`, `scripts/merge_data.py`, `data/final_dataset.csv`, `report/04_features_integration.md` |
| 5 | Modeling & report | `notebooks/05_modeling_evaluation.ipynb`, `src/`, `report/05_modeling_results.md`, `report/final_report.md`, `README.md` |

## Contract for `data/final_dataset.csv` (Member 4 → Member 5)
One row per movie. Required: `release_year`, `success` (1 if revenue ≥ $100M).
Expected features (missing ones are skipped with a warning):
`budget, budget_missing, runtime, is_sequel, release_month, num_stars, crew_size, director_avg_revenue, genre_* (one-hot), critic_score, audience_score`
Post-release (used only for the leakage demo): `vote_average, vote_count, popularity`.
**Do not** compute `director_avg_revenue` using the movie's own revenue or future movies.

## Scraped scores schema (Member 3)
`movie_id, imdb_id, critic_score, audience_score`

## Rules
1. Work on your own branch (`member1-cleaning`, ...) and open a Pull Request to `main`.
2. Commit small and often, with clear messages.
3. `git pull origin main` before you start.
4. Never commit API keys (use a local `.env`, it is git-ignored).
5. Only touch your own files, to avoid merge conflicts.
