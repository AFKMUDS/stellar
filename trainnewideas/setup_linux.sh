#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
RAP_PYTHON="${RAP_PYTHON:-python3}"
if ! "$RAP_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        "$RAP_PYTHON" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    else
        "$RAP_PYTHON" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    fi
fi
"$RAP_PYTHON" -m pip install -r requirements.txt
"$RAP_PYTHON" verify.py --quick

