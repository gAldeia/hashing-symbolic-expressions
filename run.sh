#!/bin/bash

models=("Vanilla")
models=("SimplifierBottomUp,SimplifierBottomUpOnlyLast")
#models=("SimplifierTopDown,SimplifierTopDownOnlyLast")
#models=("SimplifierTopDown1e-0,SimplifierTopDown1e-1,SimplifierTopDown1e-2,SimplifierTopDown1e-4")
#models=("SimplifierTopDown1e-6,SimplifierTopDown1e-10")

for model in "${models[@]}"
do
    # -repeats: runs per seed. -n_trials: number of different seeds
    python submit_jobs.py -repeats 1 -n_trials 50 -models "$model" -n_jobs 1 -data-dir ./data/lexicase_paper --slurm -time 00:59:00 -m 1000 # --local #
done
