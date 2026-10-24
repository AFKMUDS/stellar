# RAP Cloud Training Bundle

This bundle resumes `train_rap.py` from the included checkpoint. The JSONL
training dataset is intentionally not included: it is several gigabytes.

## Upload through JupyterLab

1. In JupyterLab's file browser, upload `stellar_rap_cloud_training.zip` to `/workspace`.
2. Open a JupyterLab terminal and extract it:

```bash
cd /workspace
python -m zipfile -e stellar_rap_cloud_training.zip .
cd cloud_train_bundle
```

3. Upload `natural_reasoning_train.jsonl` into `cloud_train_bundle/data/` through the file browser. For a large dataset, the provider's file transfer or `scp` is usually more reliable than browser upload.

## Included checkpoint

`checkpoints/b_resume_50k.pt` is the newer local checkpoint, at step 38,199.
The target step in the launch command defaults to 50,000, so that runs about
11,800 more steps. `--steps` is the final step number, not an additional-step
count.

## Prepare a cloud GPU

Use a Linux GPU instance or a CUDA-enabled PyTorch image. Install the PyTorch
build recommended by the provider for that image, then install the remaining
Python dependency:

```bash
python -m pip install -r requirements-cloud.txt
```

The bundle expects the dataset at `data/natural_reasoning_train.jsonl`. The
local dataset is at `C:\Programming\Programming_Projects\RAPextended\natural_reasoning_train.jsonl`,
but this bundle does not know a public download URL for it. If you have a
trusted dataset URL, download it on the cloud machine. Otherwise, upload it
from your PC using the provider's file transfer or `scp`:

```powershell
ssh USER@HOST "mkdir -p /workspace/cloud_train_bundle/data"
scp "C:\Programming\Programming_Projects\RAPextended\natural_reasoning_train.jsonl" USER@HOST:/workspace/cloud_train_bundle/data/
```

The JSONL records need `question` and `reference_answer` string fields.

## Start or resume

From the extracted bundle directory:

```bash
bash run_train.sh
```

The default run uses CUDA and BF16 autocast. Select a target step, data path,
or mixed-precision mode with environment variables:

```bash
STEPS=60000 bash run_train.sh
DATA=/workspace/data/natural_reasoning_train.jsonl STEPS=60000 bash run_train.sh
AMP_DTYPE=fp16 STEPS=60000 bash run_train.sh
USE_AMP=0 STEPS=60000 bash run_train.sh
```

BF16 requires GPU support; use `AMP_DTYPE=fp16` on GPUs without BF16, or
`USE_AMP=0` to disable autocast. The run script checks that both the checkpoint
and dataset exist before starting. Checkpoints are periodically saved back to
`checkpoints/b_resume_50k.pt`; keep a copy elsewhere if you need to preserve
the original.

Batch size and sequence-length settings are configurable in `run_train.sh`.
The checkpoint does not store the original data or sequence-length settings,
so adjust these only if you know the prior run's values or need to fit GPU
memory.
