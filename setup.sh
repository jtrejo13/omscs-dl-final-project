#!/usr/bin/env bash
# Setup for Vast.ai.
#
# Usage:
#   git clone <repo-url> omscs-dl-final-project && cd omscs-dl-final-project
#   bash setup.sh              # conda preferred, pip fallback
#   bash setup.sh --pip-only   # force pip-only (no conda)
#
# After setup:
#   conda activate phoenix
#   python train.py --opt experiments/train_smoke.yml
set -euo pipefail

HF_REPO="cdtrejo/sidd-medium-lmdb"
DATA_DIR="data/SIDD"
CONDA_ENV_NAME="phoenix"
USE_PIP_ONLY=false

for arg in "$@"; do
  case $arg in
    --pip-only) USE_PIP_ONLY=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

section() {
   echo
   echo "--------------------------------------"
   echo "  $1"
   echo "--------------------------------------"
}

# Python environment
section "Step 1/3: Python environment"

if [ "$USE_PIP_ONLY" = false ] && command -v conda &>/dev/null; then
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        conda env update -n "$CONDA_ENV_NAME" -f environment.yml --prune
    else
        conda env create -f environment.yml
    fi
    CONDA_PREFIX=$(conda run -n "$CONDA_ENV_NAME" python -c "import sys; print(sys.prefix)")
    PYTHON="$CONDA_PREFIX/bin/python"
    echo "  To activate: conda activate $CONDA_ENV_NAME"
else
    pip install -r requirements.txt
    PYTHON="python"
fi

# SIDD LMDB data
section "Step 2/3: SIDD LMDB dataset"

all_present=true
for d in "$DATA_DIR/train/input_crops.lmdb" "$DATA_DIR/train/gt_crops.lmdb" \
          "$DATA_DIR/val/input_crops.lmdb"   "$DATA_DIR/val/gt_crops.lmdb"; do
    if [ ! -f "$d/data.mdb" ]; then
        all_present=false
        break
    fi
done

if [ "$all_present" = true ]; then
    echo "All SIDD LMDBs present; skipping download."
else
    echo "Downloading ~70 GB from HuggingFace ($HF_REPO)..."
    mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
    
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    HF_REPO="$HF_REPO" DATA_DIR="$DATA_DIR" "$PYTHON" - <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id=os.environ["HF_REPO"],
        repo_type="dataset",
        local_dir=os.environ["DATA_DIR"],
        allow_patterns=["train/*", "val/*"],  # skip raw/ (~10 GB zip, not needed)
    )
    print("Download complete.")
except Exception as e:
    print(f"ERROR downloading dataset: {e}")
    sys.exit(1)
PYEOF
fi

for d in "$DATA_DIR/train/input_crops.lmdb" "$DATA_DIR/train/gt_crops.lmdb" \
          "$DATA_DIR/val/input_crops.lmdb"   "$DATA_DIR/val/gt_crops.lmdb"; do
    if [ ! -f "$d/meta_info.txt" ]; then
        echo "ERROR: $d/meta_info.txt missing"
        exit 1
    fi
done
echo "  LMDB validation passed."

# Smoke test
section "Step 3/3: Smoke test"
"$PYTHON" train.py --opt experiments/train_smoke.yml
echo "Smoke test passed."

section "Setup complete"
echo "  Next: conda activate $CONDA_ENV_NAME"
echo "  Then: python train.py --opt experiments/train_baseline.yml"
