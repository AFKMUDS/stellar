# Bounded Mixed-Data Pilot (not a 100K launch)

The current fresh checkpoint stays untouched. `--initialize-from` requires a
**fresh-format** checkpoint with the exact same model configuration and creates
a new optimizer, RNG, data cursor, carrier and 256-slot cache in a new run
directory. Original B weights cannot initialize this architecture. `--resume`
instead restores one existing mixed run, including its stream and carrier.

`mix_plan.example.json` sets the proposed **35/25/20/10/10 target supervised
raw-byte ratio**, not equal record counts. Copy it to an untracked plan and
fill the actual Natural Reasoning path, immutable 40-hex-character Hugging Face
revisions, selected FineWeb-Edu sample and SmolTalk2 SFT config, source licenses,
and scanned/byte budgets. The template deliberately contains placeholders and
**will fail until completed**. Schema and license of remote subsets have not
been verified in this environment. Do not rely on an unreviewed license before
publishing artifacts. Install the optional Hugging Face `datasets` dependency
in the RunPod environment, without replacing its CUDA-enabled PyTorch.

The preparer streams only the requested bounded number of records; local
JSONL inputs work without network. It adapts FineWeb to raw text, Natural
Reasoning to QA, selected SmolTalk to assistant-supervised messages, and
OpenThoughts to QA with **explicit final answers** when reasoning is off.
Long answers and texts are censored: no false EOS is trained at a cutoff.
Sources that cannot fill their byte quota fail rather than silently alter
weights. An SQLite deduplicator removes cross-source duplicates; hash-based
validation and synthetic train/val values are disjoint. The mix manifest
includes dataset revisions, user-supplied license statements, SHA256 checksums,
counts, actual supervised bytes and source weights. Remote access only occurs
if `--allow-network` is passed explicitly.

Run from the cloned repo root on RunPod, after verifying `DATA` and `SOURCE`
paths and filling the plan:

```bash
cd /workspace/stellar
python -c 'import torch; assert torch.cuda.is_available()'
python -m pip install 'datasets>=2.19,<5'
cp rap_training_rebuild/mix_plan.example.json /workspace/mix_plan.json
nano /workspace/mix_plan.json
# Replace every placeholder and review the actual subset schema/license before proceeding.
python -m rap_training_rebuild.mix \
  --plan /workspace/mix_plan.json --out /workspace/rap_mix_v1 --allow-network

SOURCE=/workspace/stellar/trainnewideas/runs/vnext/latest.pt
test -f "$SOURCE" && python -c 'import torch; assert torch.cuda.is_available()'
python -m rap_training_rebuild.vnext \
  --core fresh --primitive-mode fresh --encoding bitviews \
  --head expression --retrieval ordered --memory-readout joint \
  --state-persistence stream --mix /workspace/rap_mix_v1 \
  --config trainnewideas/configs/cloud.json \
  --prepared /workspace/rap_mix_prepared_v1 --run /workspace/rap_mixed_stage_v1 \
  --initialize-from "$SOURCE" --steps 3 --batch-size 8 \
  --chunk-length 128 --gradient-accumulation 1 --device cuda \
  --validation-target-tokens 256 --evaluate-every 1
```

The source file's architecture must match the cloud preset. The initial
mixed run prepares a disk-backed corpus, then trains three updates. Check
`/workspace/rap_mixed_stage_v1/initialization.json`, `environment.json`,
`metrics.jsonl`, `validation.jsonl` (including `by_source`), `latest.pt` and
`best.pt`. `source_supervised_bytes` reports sampled **raw** target bytes,
`target_raw_bytes` and `supervised_bytes_per_second` report training exposure
and throughput, and `peak_vram_bytes` reports allocation in that process.
Transition and expression routes are logged separately, as are retrieval
usage and per-source held-out loss/byte accuracy. Only primary next-token
loss is enabled through this launcher.

To test exact same-stage resume to six *total* steps, use the same flags,
paths and batch size, but replace `--initialize-from "$SOURCE" --steps 3`
with `--resume --steps 6`. Do not pass the original source checkpoint to
`--resume`. Extending a run changes the LR schedule's target horizon;
for serious matched experiments start with the planned target steps.

```bash
python -m rap_training_rebuild.vnext \
  --core fresh --primitive-mode fresh --encoding bitviews \
  --head expression --retrieval ordered --memory-readout joint \
  --state-persistence stream --mix /workspace/rap_mix_v1 \
  --config trainnewideas/configs/cloud.json \
  --prepared /workspace/rap_mix_prepared_v1 --run /workspace/rap_mixed_stage_v1 \
  --resume --steps 6 --batch-size 8 --chunk-length 128 \
  --gradient-accumulation 1 --device cuda \
  --validation-target-tokens 256 --evaluate-every 1
```

To compare batch sizes, create **separate** run directories such as
`/workspace/bench_b8` and `/workspace/bench_b16`, pass `--initialize-from
"$SOURCE"` and `--steps 3` for each, and change only `--batch-size 8` to
`--batch-size 16`. Read each run's `target_raw_bytes`,
`supervised_bytes_per_second` and `peak_vram_bytes` from `metrics.jsonl`.
Neither benchmark modifies `SOURCE`. On a 24 GB RTX 4090, check peak
allocation before increasing chunk length; a local 1x16 test is not a
prediction of batch-16 memory use.

```bash
for B in 8 16; do
  python -m rap_training_rebuild.vnext \
    --core fresh --primitive-mode fresh --encoding bitviews \
    --head expression --retrieval ordered --memory-readout joint \
    --state-persistence stream --mix /workspace/rap_mix_v1 \
    --config trainnewideas/configs/cloud.json \
    --prepared /workspace/rap_mix_prepared_v1 --run "/workspace/bench_b${B}" \
    --initialize-from "$SOURCE" --steps 3 --batch-size "$B" \
    --chunk-length 128 --gradient-accumulation 1 --device cuda \
    --validation-target-tokens 256 --evaluate-every 3
done
python - <<'PY'
import json
from pathlib import Path
for batch in (8, 16):
    path = Path(f"/workspace/bench_b{batch}/metrics.jsonl")
    data = json.loads(path.read_text().splitlines()[-1])
    print(batch, data["supervised_bytes_per_second"], data["peak_vram_bytes"])
PY
```

Compare the previous natural-only checkpoint and the mixed checkpoint on
the **same** held-out prepared corpus:

```bash
python -m rap_training_rebuild.compare_stage \
  --checkpoints "$SOURCE" /workspace/rap_mixed_stage_v1/latest.pt \
  --prepared /workspace/rap_mix_prepared_v1 --device cuda
```

This prints parameter counts, target-byte exposures, per-source loss/accuracy,
routing/retrieval metrics and greedy generations from the same prompt and
seed. It does **not** certify equal exposure if the target-byte counts differ.
For free-generation exact recall and reversible-byte probes on the fixed
synthetic validation source:

```bash
python -m rap_training_rebuild.probes \
  --checkpoint /workspace/rap_mixed_stage_v1/latest.pt \
  --validation /workspace/rap_mix_v1/synthetic.validation.jsonl \
  --device cuda --count 12
```

Also verify that the natural validation records were not in the old run's
training split; a new hash split alone cannot prove that. Do not launch a
5K/100K campaign until that comparison, the RunPod smoke and data provenance
are reviewed.

If those gates pass, launch a **separate** 5,000-update candidate run (do not
resume the six-step smoke run, which has a different schedule):

```bash
python -m rap_training_rebuild.vnext \
  --core fresh --primitive-mode fresh --encoding bitviews \
  --head expression --retrieval ordered --memory-readout joint \
  --state-persistence stream --mix /workspace/rap_mix_v1 \
  --config trainnewideas/configs/cloud.json \
  --prepared /workspace/rap_mix_prepared_v1 --run /workspace/rap_mixed_pilot_b8 \
  --initialize-from "$SOURCE" --steps 5000 --batch-size 8 \
  --chunk-length 128 --gradient-accumulation 1 --device cuda \
  --validation-target-tokens 4096 --evaluate-every 250
```

Resume that exact pilot with the same flags and paths, replacing
`--initialize-from "$SOURCE"` with `--resume` and keeping `--steps 5000`.
At batch 8 versus an old batch-1 checkpoint, 5,000 updates are **not** an
equal supervised-byte budget: compare the recorded raw target bytes, not the
update count.
