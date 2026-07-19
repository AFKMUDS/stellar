$ErrorActionPreference = "Stop"

$Common = @(
    "--data", "natural_reasoning_train.jsonl",
    "--steps", "2500",
    "--batch-size", "8",
    "--seq-len", "512",
    "--max-prompt", "384",
    "--max-target", "192",
    "--lr", "0.0002",
    "--seed", "17",
    "--device", "cuda",
    "--amp",
    "--amp-dtype", "bf16",
    "--save-every", "500",
    "--metrics-every", "100",
    "--use-symmetry-retrieval",
    "--switch-cost-end", "1250",
    "--balance-end", "500",
    "--phase1-end", "250",
    "--phase2-end", "1250",
    "--release-end", "2000"
)

python train_rap_persistent2.py --experiment A --out RobinWilliams/A/rap8.pt @Common
python train_rap_persistent2.py --experiment B --out RobinWilliams/B/rap2.pt @Common
python train_rap_persistent2.py --experiment C --out RobinWilliams/C/persistent2.pt @Common
python train_rap_persistent2.py --experiment D --out RobinWilliams/D/persistent2_staged.pt @Common
