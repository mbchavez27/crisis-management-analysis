#!/usr/bin/env bash
# Fresh-PC setup: env + Tagalog spaCy model + NLTK data.
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "=== 1/3 uv sync ==="
uv sync

echo "=== 2/3 Tagalog spaCy model (tl_calamancy_md, not on PyPI) ==="
# The upstream wheel filename carries no version, which modern pip/uv reject.
# Download it, re-tag with its internal METADATA version, then install.
# Skipped when the model already imports (uv sync wipes unmanaged packages,
# so this step must run after every sync on a fresh venv).
if uv run python -c "import spacy; spacy.load('tl_calamancy_md')" 2>/dev/null; then
  echo "tl_calamancy_md already installed, skipping."
else
  MODEL_URL="https://huggingface.co/ljvmiranda921/tl_calamancy_md/resolve/main/tl_calamancy_md-any-py3-none-any.whl"
  WHEEL_FILE=$(uv run python - "$MODEL_URL" <<'EOF'
import sys, urllib.request, zipfile, tempfile, os
url = sys.argv[1]
tmp = os.path.join(tempfile.gettempdir(), "tl_calamancy_md.whl")
urllib.request.urlretrieve(url, tmp)
with zipfile.ZipFile(tmp) as z:
    meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    ver = next(l.split(":", 1)[1].strip() for l in z.read(meta).decode().splitlines() if l.startswith("Version:"))
tagged = os.path.join(tempfile.gettempdir(), f"tl_calamancy_md-{ver}-py3-none-any.whl")
os.replace(tmp, tagged)
print(tagged)
EOF
)
  uv pip install --no-deps "$WHEEL_FILE"
  uv run python -c "import spacy; spacy.load('tl_calamancy_md'); print('tl_calamancy_md ok')"
fi

echo "=== 3/3 NLTK data ==="
uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "Done. Next:"
echo "  cp .env.example .env   # then fill GEMINI_API_KEY / GEMMA_API_KEY"
echo "  uv run python main.py --stage pipeline"
