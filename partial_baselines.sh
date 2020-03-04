python3 -m ActiveDialogue.main.partial \
  --strategy $1 \
  --label_budget 20000 \
  --pool_size 5000 \
  --al_batch 128 \
  --batch_size 64 \
  --seed_batch_size 64 \
  --comp_batch_size 32 \
  --inference_batch_size 128 \
  --gamma 0.5 \
  --eval_period 1 \
  --seed 4000 \
  --seed_size 50 \
  --epochs 5 \
  --seed_epochs 500 \
  --model glad \
  --device 0 \
  --lr 0.001 \
  --threshold_strategy $2 \
  --init_threshold 0.2 \
  --threshold_scaler 0.00005 \
  --rejection_ratio 32 \