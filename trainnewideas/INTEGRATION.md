# Recovered Fresh Core: Integration Notes

`RAP_Fresh_Training_recovered_core.zip` supplied `rap/`, `train.py`,
`configs/cloud.json`, and an identical `verify.py`. They are kept separately
from `raplib/` and original checkpoint B. The old and fresh weights have
different shapes and vocabularies; no automatic migration is performed.

From the repo root:

```bash
python trainnewideas/verify.py
python -m rap_training_rebuild.vnext --help
```

The 13 CPU verification tests pass on the local CUDA-enabled PyTorch
installation. A separate local RTX 4070 float32 smoke used the full 6,295,912
parameter cloud model for two training steps, resumed to step three, and logged
held-out language loss plus independent transition/expression counts. Peak
allocated GPU memory for that 1x16-symbol smoke was 243,846,144 bytes; this
does **not** predict memory use for cloud chunk_length=256 / batch_size=8.
Validation included 16 supervised targets and EOS, so bits-per-byte uses
only the 11 raw-byte targets. The successful smoke does not establish language
quality or completion of the whole vNext research protocol.

Use `python -m rap_training_rebuild.vnext --core fresh ...` for the supported
flagged path. It refuses unsupported feature combinations before preparing
data. `--core original` uses only the existing small B-checkpoint pilot, not
the disk-backed fresh pipeline: the 500 MB corpus needs the original
`train_rap.py` (see `rap_training_rebuild/README.md`) until a genuinely matched
original data path is implemented. Minimal `configs/laptop.json` and
`requirements.txt` were added to make the recovered launchers runnable.
The recovered bundle still lacks `infer.py`, `diagnose.py`, and
`CHAT_WINDOWS.bat`; do not follow commands referencing those missing files.
