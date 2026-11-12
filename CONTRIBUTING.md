# Team Workflow

## Who uploads what
| Person | Role | Upload to |
|---|---|---|
| 1 | Web scraping | `src/scraping.py`, `data/scraped_scores.csv` |
| 2 | Cleaning & parsing | `src/cleaning.py`, `notebooks/cleaning.ipynb` |
| 3 | Merging & features | `src/features.py`, `data/final_dataset.csv` |
| 4 | EDA & visualization | `notebooks/eda.ipynb`, `results/figures/` |
| 5 | Modeling & README | `src/modeling.py`, `README.md`, `docs/` |

## Agreed schema for scraped scores
`movie_id, imdb_id, critic_score, audience_score`

## Rules
1. Never push directly to `main`. Work on your own branch: `person1-scraping`, `person2-cleaning`, ...
2. Commit small and often, with clear messages (e.g. `Add metacritic scraper`).
3. Open a Pull Request to `main`; one teammate reviews it before merge.
4. Pull before you start working: `git pull origin main`.
5. Never commit API keys. Put them in a local `.env` file (it is git-ignored).
6. Only touch your own files, to avoid merge conflicts.
