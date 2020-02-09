# python3 -m ActiveDialogue.main.$1 --label_budget=1000000 --seed 34 --al_batch 4096 --fit_items 2048 --batch_size 128 --eval_period 2 --recency_bias 0
python3 -m ActiveDialogue.main.$1 --label_budget=1000000 --al_batch 4096 --fit_items 4096 --batch_size 128 --eval_period 1 --recency_bias 0 --seed 9 --seed_size 0 --epochs 1 --device 0
# python3 -m ActiveDialogue.main.$1 --label_budget=10000 --seed 34 --al_batch 8 --fit_items 4 --batch_size 16 --eval_period 2 --recency_bias 0 --seed_size 2 --device 0 --eval_period 10 --model glad --gamma 1
