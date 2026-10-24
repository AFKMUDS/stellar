#!/usr/bin/env bash
set -euo pipefail

DATA="${DATA:-data/natural_reasoning_train.jsonl}"
OUT="${OUT:-checkpoints/b_resume_50k.pt}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-768}"
MAX_PROMPT="${MAX_PROMPT:-512}"
MAX_TARGET="${MAX_TARGET:-256}"
LR="${LR:-0.0002}"

if [[ ! -f "$OUT" ]]; then
  echo "Checkpoint not found: $OUT" >&2
  echo "Set OUT to an existing checkpoint path to resume." >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "Training dataset not found: $DATA" >&2
  echo "Set DATA to the JSONL path containing question/reference_answer records." >&2
  exit 1
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Use a CUDA-enabled PyTorch environment and GPU.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
PY

AMP_ARGS=()
if [[ "${USE_AMP:-1}" != "0" ]]; then
  AMP_ARGS=(--amp --amp_dtype "${AMP_DTYPE:-bf16}")
fi

python train_rap.py \
  --data "$DATA" \
  --out "$OUT" \
  --resume \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --max_prompt "$MAX_PROMPT" \
  --max_target "$MAX_TARGET" \
  --lr "$LR" \
  --device cuda \
  --use_symmetry_retrieval \
  --use_numeric_features \
  --use_aux_numeric \
  "${AMP_ARGS[@]}"
