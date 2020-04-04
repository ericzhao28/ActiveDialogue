# Train seed
# python3 -m ActiveDialogue.main.vanilla \
#   --strategy lc \
#   --label_budget 5000 \
#   --pool_size 5000 \
#   --al_batch 256 \
#   --batch_size 64 \
#   --seed_batch_size 64 \
#   --comp_batch_size 32 \
#   --inference_batch_size 256 \
#   --gamma 0.3 \
#   --eval_period 1 \
#   --seed 4000 \
#   --seed_size 50 \
#   --epochs 20 \
#   --seed_epochs 500 \
#   --model glad \
#   --device 0 \
#   --lr 0.001 \
#   --force_seed

# LC 
python3 -m ActiveDialogue.main.vanilla \
  --pool_size 5000 \
  --seed_size 50 \
  --label_budget 5000 \
  --al_batch 256 \
  --batch_size 64 \
  --seed_batch_size 64 \
  --comp_batch_size 32 \
  --inference_batch_size 256 \
  --strategy lc \
  --gamma 0.3 \
  --eval_period 1 \
  --seed 1 \
  --epochs 20 \
  --seed_epochs 500 \
  --model glad \
  --device 0 \
  --lr 0.001 \
  --noise_fp 0.0 \
  --noise_fn 0.0 \
  --threshold_strategy fixed \
  --init_threshold 0.2
