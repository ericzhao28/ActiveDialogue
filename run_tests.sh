python3 -m tests.trivial_train --model gce --lr 0.1 --epochs 3 --seed 30300 --al_batch 4 --pool_size 30 --label_budget 9 --fit_items 16 --seed_size 64 --seed_epochs 2 --batch_size 2
python3 -m tests.trivial_train --model glad --lr 0.1 --epochs 3 --seed 30301 --al_batch 4 --pool_size 15  --label_budget 9 --fit_items 16 --seed_size 64 --seed_epochs 2 --batch_size 2
python3 -m tests.random_dst_env --model glad --lr 0.1 --epochs 3 --seed 30301 --al_batch 4 --pool_size 15  --label_budget 9 --fit_items 16 --seed_size 64 --seed_epochs 2 --batch_size 2 --device 0
python3 -m tests.dst_env --lr 0.1 --epochs 3 --seed 30301 --al_batch 4 --pool_size 15  --label_budget 9 --fit_items 16 --seed_size 64 --seed_epochs 2 --batch_size 2 --device 0
python3 -m tests.naive_baselines
python3 -m tests.dataset
python3 -m tests.toy_e2e_test
