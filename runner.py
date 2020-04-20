from joblib import Parallel, delayed
import queue
import os
import time

GPU_available = [0, 1]
N_PARALLEL = 1
N_GPU = len(GPU_available)

##################################################
# Base
vanilla_params = "--model glad --device 0 --seed 2 "
vanilla_seed_cmd = "python3 -m ActiveDialogue.main.seed --lr 0.001 --force_seed --seed_epochs 300 " + vanilla_params
vanilla_cmd = "python3 -m ActiveDialogue.main.vanilla  " + vanilla_params

experiments = [
    vanilla_cmd + " --strategy bald --init_threshold 0.4 ",
    vanilla_cmd + " --strategy entropy --init_threshold 10.0 ",
    vanilla_cmd + " --strategy aggressive ",
    vanilla_cmd + " --strategy passive ",
]
print(experiments)
##################################################

# Put indices in queue
q = queue.Queue(maxsize=N_GPU * N_PARALLEL)
for j in range(N_PARALLEL):
    for i in range(N_GPU):
        q.put(i)


def runner(cmd):
    gpu = q.get()
    os.system(cmd + " --device %d " % gpu)
    q.put(gpu)


# Change loop
Parallel(n_jobs=N_GPU * N_PARALLEL,
         backend="multiprocessing")(delayed(runner)(e) for e in experiments)
