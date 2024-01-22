#!/bin/bash

models=()

#models+=("Vanilla")

models+=("SimplifierBottomUp")
models+=("SimplifierTopDown")

models+=("SimplifierBottomUpOnlyLast")
models+=("SimplifierTopDownOnlyLast")

models+=("SimplifierTopDown1e-0,SimplifierTopDown1e-1")
models+=("SimplifierTopDown1e-2,SimplifierTopDown1e-4")
models+=("SimplifierTopDown1e-6,SimplifierTopDown1e-10")

models+=("SimplifierBottomUp1e-0,SimplifierBottomUp1e-1")
models+=("SimplifierBottomUp1e-2,SimplifierBottomUp1e-4")
models+=("SimplifierBottomUp1e-6,SimplifierBottomUp1e-10")

for model in "${models[@]}"
do
    # -repeats: runs per seed. -n_trials: number of different seeds
    python submit_jobs.py -repeats 1 -n_trials 30 -models "$model" -n_jobs 60 -data-dir ./data/lexicase_paper --local # --slurm -time 1:45:00 -m 1000 # 
done
