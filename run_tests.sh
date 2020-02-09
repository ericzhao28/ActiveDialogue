python3 -m tests.dst_env
python3 -m tests.naive_baselines
python3 -m tests.dataset
python3 -m tests.trivial_train --model gce --lr 0.1 --epochs 3
python3 -m tests.trivial_train --model glad --lr 0.1 --epochs 3
