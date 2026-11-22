#!/usr/bin/env bash
# Run once from the project root:  bash setup_repo.sh
# Creates the folder layout and moves the notebook. It does NOT touch git history.
set -e

mkdir -p data models reports notebooks tests dev
touch data/.gitkeep models/.gitkeep reports/.gitkeep

if [ -f final_project.ipynb ]; then
  mv final_project.ipynb notebooks/
  echo "moved final_project.ipynb -> notebooks/"
fi

# Safety checks before the first push
if [ -f .env ] && ! git check-ignore -q .env; then
  echo "WARNING: .env is NOT ignored - add .gitignore before committing!"; exit 1
fi
if git rev-parse --git-dir >/dev/null 2>&1 && [ -n "$(git log --all --oneline -- .env 2>/dev/null)" ]; then
  echo "WARNING: .env exists in git history. Rotate your TMDB/OMDb keys."
fi
echo "Layout ready. Next: git status  (check that .env and data/*.csv are NOT listed)"
