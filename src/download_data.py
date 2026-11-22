import shutil
import kagglehub
import config

path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
for name in ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv"):
    shutil.copy(f"{path}/{name}", config.DATA_DIR / name)
    print("Copied", name)
