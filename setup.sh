#!/usr/bin/env bash
# Fresh-PC setup: one uv env per stage + Tagalog spaCy model + NLTK data.
# 03-llm-passes (vLLM, GPU only) is set up separately, see README.
# Usage: bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "=== 1/3 uv sync (01, 02, 04, 05) ==="
for stage in 01-preprocessing 02-analysis 04-statistical-analysis 05-summary; do
  echo "--- $stage ---"
  uv sync --project "$stage"
done

echo "=== 2/3 Tagalog spaCy model (tl_calamancy_md, not on PyPI) into 02-analysis ==="
# The upstream wheel filename carries no version, which modern pip/uv reject.
# Download it, re-tag with its internal METADATA version, then install.
# Skipped when the model already imports (uv sync wipes unmanaged packages,
# so this step must run after every sync on a fresh venv).
if uv run --project 02-analysis python -c "import spacy; spacy.load('tl_calamancy_md')" 2>/dev/null; then
  echo "tl_calamancy_md already installed, skipping."
else
  MODEL_URL="https://huggingface.co/ljvmiranda921/tl_calamancy_md/resolve/main/tl_calamancy_md-any-py3-none-any.whl"
  WHEEL_FILE=$(uv run --project 02-analysis python - "$MODEL_URL" <<'EOF'
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
  uv pip install --python 02-analysis/.venv --no-deps "$WHEEL_FILE"
  uv run --project 02-analysis python -c "import spacy; spacy.load('tl_calamancy_md'); print('tl_calamancy_md ok')"
fi

echo "=== 3/3 NLTK data (shared ~/nltk_data, used by 01 and 02) ==="
uv run --project 01-preprocessing python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"

echo ""
echo "Done. Next:"
echo "  cp .env.example .env   # then fill GEMINI_API_KEY"
echo "  python3 main.py --stage pipeline"
