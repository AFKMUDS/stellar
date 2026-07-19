$ErrorActionPreference = "Stop"

python train_rap_persistent2.py `
  --experiment D `
  --data natural_reasoning_train.jsonl `
  --out RobinWilliams/D/persistent2_staged.pt `
  --steps 20000 `
  --batch-size 8 `
  --seq-len 512 `
  --max-prompt 384 `
  --max-target 192 `
  --lr 0.0002 `
  --seed 17 `
  --device cuda `
  --amp `
  --amp-dtype bf16 `
  --save-every 1000 `
  --metrics-every 100 `
  --use-symmetry-retrieval `
  --switch-cost 0.01 `
  --switch-cost-end 9000 `
  --balance-weight 0.01 `
  --balance-end 3000 `
  --carrier-target 0.8 `
  --phase1-end 2000 `
  --phase2-end 9000 `
  --release-end 14000
