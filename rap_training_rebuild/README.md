# RAP checkpoint-B training rebuild (pilot)

This folder isolates the proposed readout-versus-memory comparison from existing RAP scripts.
The base is **`rap_B_latest (1).pt` (saved step 87,199)**, not `rap_B_current.pt`
(saved step 58,999). No checkpoint or Qwen weights are modified in place.

## vNext integration status

The recovered `trainnewideas/` directory now has the fresh `rap/` core,
`configs/cloud.json`, `train.py`, and a working `verify.py`. The unified guarded
launcher `python -m rap_training_rebuild.vnext` selects either the original
checkpoint-B pilot or the separately trained fresh implementation. The
`experiment.py` CLI also accepts the proposed flag names so unsupported choices
(fresh core/primitives, bitviews/multi encoding, expression head/loss,
ordered/random/disabled retrieval, clock, router losses, stream persistence,
curriculum, full diagnostics) fail *before* training rather than silently
falling back on the **original** core. For the fresh core, `vnext` supports
fresh primitives, bitviews, direct/expression head, ordered/random/none
retrieval, the fresh joint memory readout, and stream persistence. All other
proposed features still fail before training. `--head direct` is supported
only for the fresh core.

The original forward path has an exact parity test on the local B checkpoint.
The fresh core has disk-backed large-data preparation, tokens above 255,
ordered bounded retrieval, reversible maps, persistent chunk state and exact
new-format resume. It passed 13 CPU verification tests and a short CUDA
train/resume test on the local RTX 4070. This is **not** full vNext readiness:
cross-core mixed feature combinations, expression displacement, learned clock,
and full intervention diagnostics are not implemented. The original and fresh
vocabularies and supervision formats differ; do not claim strict matched-data
A-G comparisons or launch a long campaign without those controls.
`load_examples` handles `prompt`/`answer`, `question`/`reference_answer`,
`instruction`/`output`, and `text` JSONL records for small experiments; this
pilot eagerly loads JSONL, so **do not point it at the 500 MB corpus**.

From the repo root on Runpod, for a **fresh-core preparation + short smoke**
using the uploaded dataset, set `DATA` to its actual path:

```bash
DATA=/workspace/data/natural_reasoning_train.jsonl
python -m rap_training_rebuild.vnext --core fresh --primitive-mode fresh \
  --encoding bitviews --head expression --retrieval ordered \
  --memory-readout joint --state-persistence stream --dataset "$DATA" \
  --prepared trainnewideas/data/prepared --run trainnewideas/runs/vnext \
  --config trainnewideas/configs/cloud.json --steps 3 --device cuda \
  --batch-size 1 --chunk-length 128 --gradient-accumulation 1
```

For a **fresh-format checkpoint only**, resume the same prepared data and
configuration with an absolute target step (do not pass an original B
checkpoint as `--initialize-from`):

```bash
python -m rap_training_rebuild.vnext --core fresh --primitive-mode fresh \
  --encoding bitviews --head expression --retrieval ordered \
  --memory-readout joint --state-persistence stream \
  --prepared trainnewideas/data/prepared --run trainnewideas/runs/vnext \
  --config trainnewideas/configs/cloud.json --steps 6 --resume --device cuda \
  --batch-size 1 --chunk-length 128 --gradient-accumulation 1
```

Fresh outputs never overwrite `rap_training_rebuild/outputs/` or the
original `rap_B_latest (1).pt`. Both runs explicitly set the extra fresh
auxiliary losses to zero; only next-token loss is optimized. This recovered
bundle lacks the original release's `infer.py` and `diagnose.py`; use the
cloud preset and the tested `rap.runtime.Session` /
`rap.evaluation.evaluate` APIs instead. See `trainnewideas/INTEGRATION.md`.

For a full-corpus **original** baseline on Runpod, use the existing original
trainer until a matched disk-backed original pilot is available. From the cloned
repo root on Runpod (adjust `DATA` to the actual existing location):

```bash
cd cloud_train_bundle
DATA=data/natural_reasoning_train.jsonl OUT=checkpoints/b_resume_50k.pt STEPS=38500 bash run_train.sh
```

This is a ~300-update **original-core smoke run** from the included step-38,199
checkpoint. `STEPS` is an absolute target, not extra steps. The script checks
the checkpoint and dataset exist and refuses CUDA-less environments; it
automatically enables the original symmetry retrieval and numeric features.
It does **not** activate any vNext experimental flags. The full dataset and
87,199-step checkpoint are intentionally not stored in Git. If either path
on Runpod differs, set `DATA` or `OUT` to that exact path; do not assume
`b_resume_50k.pt` is the 87,199-step checkpoint.

## Four matched conditions

| Variant | Core | Byte readout |
| --- | --- | --- |
| A | original, frozen | original |
| B | original, frozen | new state + embedding + symmetry-retrieval MLP |
| C | fine-tuned | original linear head, fine-tuned |
| D | fine-tuned | new state + embedding + symmetry-retrieval MLP |

The runner uses RAP's exact symmetry-retrieval rule and a detached state history,
as in `raplib/model.py`; it preserves full trajectories (no artificial frozen-state
shortcuts). B trains only its new readout. C and D train the recurrent core; C
also updates the native head. This is a **small four-way pilot**, not an implemented
30,000-step multi-stage curriculum. It does not add new retrieval keys, or
demonstrate conversation competence by itself.

## Data

Optional: generate two-turn practice dialogues with *local* `Qwen/Qwen3-0.6B`:

```bash
python -m rap_training_rebuild.make_conversations
```

Requires `transformers`, the Qwen3-0.6B weights/tokenizer already cached locally,
and adequate memory (prefer a GPU). `local_files_only=True` prevents downloads.
If not cached, provision Qwen3-0.6B separately with your approval before running.
The repo's `qwen2.5-0.5b-instruct-q4_k_m.gguf` is **not** Qwen3-0.6B.
Teacher responses are plain supervised UTF-8 examples, **not** Qwen token-logit
distillation. Inspect the generated `conversations.jsonl` for answer quality.

The pilot creates train/validation memory and operations examples automatically:
disjoint keys/values/operation inputs, delay buckets, distractors, overwritten
keys, and unknown keys. Val uses distinct delays. For general-language and
stability examples, provide optional JSONL files containing
`{"prompt": "...", "answer": "..."}`; conversation rows also support `group`
to prevent turns from the same dialogue crossing splits. A stable hash reserves
10% of groups for validation. Supply sufficiently many examples to populate
both splits. Do not reuse prompts or data from validation in another category.

Sampling targets are 35% language, 30% conversation, 20% memory, 10% operations,
5% stability. Missing optional categories are removed and the remaining weights
renormalized. Thus **without Qwen JSONL, the run is not a conversation experiment**.

## Run

Run from the repo root with the same data and evaluation flags for all variants.
For example, after generating conversations:

```bash
python -m rap_training_rebuild.experiment --variant A --conversations rap_training_rebuild/conversations.jsonl --max-delay 256 --seq-len 2048
python -m rap_training_rebuild.experiment --variant B --conversations rap_training_rebuild/conversations.jsonl --max-delay 256 --seq-len 2048 --steps 3000
python -m rap_training_rebuild.experiment --variant C --conversations rap_training_rebuild/conversations.jsonl --max-delay 256 --seq-len 2048 --steps 3000
python -m rap_training_rebuild.experiment --variant D --conversations rap_training_rebuild/conversations.jsonl --max-delay 256 --seq-len 2048 --steps 3000
```

Use `--language path.jsonl --stability path.jsonl` to include the other categories.
To compare long-delay recall, increase `--max-delay 4096 --seq-len 4600` on
all variants; this costs substantially more memory/time. `--batch-size`,
`--seed`, `--lr`, `--device`, `--eval-examples`, and `--rollouts` are configurable.
Training refuses to silently truncate examples. Outputs default to
`rap_training_rebuild/outputs/{variant}.pt`; `--resume` restores optimizer and
sampler RNG, while `--eval-only` loads a trained B/C/D model for evaluation.
Do not change data, lengths, or seed across a resumed comparison.

Evaluation prints held-out answer-byte loss and accuracy, teacher-forced exact
answers, free-generation exact prefixes, and a simple repetition/tab indicator.
Memory is separated by delay. This small benchmark is **not** an independent
general-language corpus evaluation, full primitive-ablation suite, or proof of
4K retrieval-to-speech; add these before selecting a production checkpoint.
The Qwen generation and all long runs require explicit compute provisioning.

## Optional online Qwen teacher

Instead of reusing only the 20 fixed examples, add `--online-teacher` to a
**training** B/C/D run. Every `--teacher-every 50` optimization steps (including
the first), RAP is probed on a small **development** memory set with disjoint
keys/values/delays. The weakest dev delay chooses the next training delay; Qwen
generates a fresh neutral memory passage and a separate conversation question
and answer. The memory answer is computed from the binding, **not** trusted to
Qwen. Qwen answers for conversation are format-, length-, duplicate-, and
first-person-filtered, but may still contain factual errors. Inspect the teacher
JSONL before relying on longer runs. Invalid examples are rejected, not silently
trained on.

```bash
python -m rap_training_rebuild.experiment --variant B \
  --conversations rap_training_rebuild/conversations.jsonl \
  --online-teacher --teacher-every 50 --teacher-device cpu \
  --max-delay 1024 --seq-len 1400 --steps 3000
```

Teacher weights are loaded only from the local Qwen3-0.6B cache; no network
fetch is attempted. RAP still trains by byte-level supervised prediction, with
no Qwen access to the router, state, or primitive parameters. The teacher sees
only the *dev loss per delay*, not evaluation prompts, weights, or hidden states.
It writes **training-only** examples to `{out-stem}.teacher.jsonl`; final
evaluation uses the pre-existing `val` split and separate synthetic validation
keys/values. Resuming requires the same output, settings, and teacher log.
An interrupted write may require a new output path rather than silently
replaying an inconsistent run.

If `outputs/B.pt` already records step 100, resume it with a **total** step
target greater than 100. In a CUDA-enabled PowerShell environment, this
continues to step 200 and shows the teacher's training challenges and RAP's
greedy replies when the teacher refreshes:

```powershell
python -m rap_training_rebuild.experiment --variant B --conversations rap_training_rebuild/conversations.jsonl --online-teacher --teacher-every 50 --teacher-device cpu --max-delay 256 --seq-len 1024 --steps 200 --batch-size 1 --device cuda --eval-examples 10 --rollouts 3 --resume --show-dialogue --rollout-bytes 64
```

`--show-dialogue` prints the latest saved training example immediately on
resume and new examples at refreshes (normally every 50 steps). The memory
answer is verified by the task generator; only conversation answers are Qwen's.
RAP's reply is a bounded greedy sample, **not** an actual live exchange with
Qwen or proof of correctness. Long memory context is abbreviated in the
terminal; the full prompt is kept in `outputs/B.teacher.jsonl`.

Online adaptation changes the training examples by variant: **do not use its
B/C/D numbers as the matched-data four-way causal comparison**. Run the fixed
data comparison above first; online teaching is a separate curriculum branch.
CPU teacher inference competes with RAP compute and may be slow; 4K delays
are significantly more expensive. The teacher does not guarantee good answers
or generalization: use held-out human-reviewed conversation sets in addition
to the fixed synthetic probes for serious claims.

Quick local check (CPU, no Qwen):

```bash
python -m unittest rap_training_rebuild.test_experiment -v
python -m rap_training_rebuild.experiment --variant A --max-delay 64 --seq-len 400 --eval-examples 1 --rollouts 0
```
