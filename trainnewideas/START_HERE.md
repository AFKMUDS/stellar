# Start here

Use your existing Miniconda environment with CUDA PyTorch.

The recovered core is now in `trainnewideas/`. For verified, guarded
original/fresh flag selection, start with `INTEGRATION.md` and
`rap_training_rebuild/README.md`. Do not load original B weights into this
fresh model.

1. Open `trainnewideas/`.
2. Run SETUP_WINDOWS.bat once in the CUDA-enabled Miniconda environment.
3. Drag ultrachat_train.jsonl (or another supported dataset) onto
   TRAIN_WINDOWS.bat.
4. Inspect `runs/rap/validation.jsonl` and `samples.jsonl`. Interactive
   `infer.py`/`CHAT_WINDOWS.bat` are not in the recovered archive.

You can instead put a dataset file directly in data/ and run TRAIN_WINDOWS.bat.

The first run prepares the dataset, creates a held-out split, and starts fresh
training. Later runs of that launcher resume the same run automatically. Use
Ctrl+C to finish the current update and save before stopping.

Your original checkpoint is not needed. This is the new model we will judge
after training, not a wrapper around B.

## If you prefer one command

From a Miniconda Prompt inside this folder:

```powershell
python train.py --dataset "C:\Programming\Programming_Projects\stellar\ultrachat_train.jsonl"
```

## On a CUDA cloud pod

With its Python environment active:

```bash
python -m pip install -r requirements.txt
python train.py --config configs/cloud.json --dataset /workspace/your_dataset.jsonl
```

Resume with:

```bash
python train.py --config runs/rap/config.json --prepared data/prepared --resume auto
```

Use a persistent /workspace folder for the extracted project, prepared corpus
and runs. A tmux session keeps the terminal job independent of your phone
connection.

## What to expect

Training prints supervised language loss and progress. Validation, sample
responses and both routing distributions are logged periodically. Only useful
behavior on your actual data can tell us how this version will speak.

The smoke configuration and verifier establish that the pipeline runs; they
are not the training run you want to keep. README.md has the controls and
new-stage commands.
