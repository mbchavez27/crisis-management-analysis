#!/usr/bin/env bash
# Fresh-PC setup: env + Tagalog spaCy model + NLTK data.
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "=== 1/3 uv sync ==="
uv sync

echo "=== 2/3 Tagalog spaCy model (tl_calamancy_md, not on PyPI) ==="
uv run python -m pip install --no-deps \
  "https://huggingface.co/ljvmiranda921/tl_calamancy_md/resolve/main/tl_calamancy_md-any-py3-none-any.whl"

echo "=== 3/3 NLTK data ==="
uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "Done. Next:"
echo "  cp .env.example .env   # then fill GEMINI_API_KEY / GEMMA_API_KEY"
echo "  uv run python main.py --stage pipeline"
