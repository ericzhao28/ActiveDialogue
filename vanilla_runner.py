from joblib import Parallel, delayed
import queue
import os
import time

# Define number of GPUs available
GPU_available = [0]
N_GPU = len(GPU_available)

experiments = [
"""
python3 -m ActiveDialogue.main.vanilla \
  --strategy bald \
  --gamma 0.7 \
  --seed 2 \
  --epochs 20 \
  --seed_epochs 100 \
  --model glad \
  --init_threshold 0.4 \
  --device 0 \
""",
"""
python3 -m ActiveDialogue.main.vanilla \
  --strategy entropy \
  --gamma 0.7 \
  --seed 2 \
  --epochs 20 \
  --seed_epochs 100 \
  --model glad \
  --init_threshold 10.0 \
  --device 0 \
""",
"""
python3 -m ActiveDialogue.main.vanilla \
  --strategy aggressive \
  --gamma 0.7 \
  --seed 2 \
  --epochs 20 \
  --seed_epochs 100 \
  --model glad \
  --device 0 \
""",
"""
python3 -m ActiveDialogue.main.vanilla \
  --strategy passive \
  --gamma 0.7 \
  --seed 2 \
  --epochs 20 \
  --seed_epochs 100 \
  --model glad \
  --device 0 \
""",
]

# Put indices in queue
q = queue.Queue(maxsize=N_GPU)
mapper = {}
invert_mapper = {}
for i in range(N_GPU):
    mapper[i] = GPU_available[i]
    invert_mapper[GPU_available[i]] = i
    q.put(i)

def runner(cmd):
    gpu = mapper[q.get()]
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))
    q.put(invert_mapper[gpu])

Parallel(n_jobs=N_GPU, backend="threading")( delayed(runner)(e) for e in experiments)
