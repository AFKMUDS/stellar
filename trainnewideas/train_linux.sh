#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
RAP_PYTHON="${RAP_PYTHON:-python3}"
RAP_TRAIN_CONFIG="${RAP_CONFIG:-configs/laptop.json}"
if [ -f runs/rap/config.json ]; then
    RAP_TRAIN_CONFIG=runs/rap/config.json
fi
exec "$RAP_PYTHON" train.py --config "$RAP_TRAIN_CONFIG" --resume auto "$@"

