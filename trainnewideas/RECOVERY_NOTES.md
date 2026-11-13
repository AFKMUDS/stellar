# Recovered RAP Fresh Training core — November 13, 2026

This bundle reconstructs the **fresh** core files separately located in the ChatGPT file library. It is intended for the coder to merge into the existing repo's `trainnewideas/`, not to replace or overwrite the original-core implementation.

## Contents
- `train.py` — fresh training entrypoint
- `rap/` — matched package (`__init__.py`, `config.py`, `data.py`, `engine.py`, `evaluation.py`, `losses.py`, `model.py`, `runtime.py`, `tokens.py`)
- `configs/cloud.json` — recovered fresh cloud preset
- `verify.py` — recovered verification suite

## Verified locally
`python verify.py` — **13/13 tests passed on CPU**, including a tiny training/generation task, deterministic resume, causal loss, cross-chunk continuity, memory and byte-map tests. Test runtime had PyTorch 2.10.0+cpu. No CUDA or target-data quality test was performed. Source files were reassembled based on package interfaces; they were not validated against the original zip's checksums.

## Required next steps
1. Extract alongside the existing original pilot; retain the original commit and back up any overlapping source.
2. Run `python verify.py` in the fresh folder.
3. Check `python train.py --help` and validate the cloud config against the desired GPU environment.
4. Port the vNext flags deliberately, keeping original-core parity and separate original/fresh training paths.
5. Only launch a costly full training job after CUDA smoke/resume and held-out dataset checks.

Auxiliary scripts, other presets, and full original docs are not included here; this package is the recovered, tested **core**, not a complete copy of every file in the original release.
