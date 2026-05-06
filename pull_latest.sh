#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="${1:-CNN---cuda}"

cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Errore: questa cartella non sembra una repository Git."
  exit 1
fi

echo "Repository: $REPO_DIR"
echo "Remote origin:"
git remote get-url origin
echo

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Errore: ci sono modifiche locali non salvate."
  echo "Salvale con commit/stash prima di fare pull:"
  echo "  git status"
  exit 1
fi

echo "Passo al branch $BRANCH..."
git checkout "$BRANCH"

echo "Scarico gli aggiornamenti..."
git pull --ff-only origin "$BRANCH"

echo
echo "Pull completato."
