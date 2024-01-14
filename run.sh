#!/bin/bash

models=("SimplifierBottomUp,SimplifierTopDown,Vanilla")
models=("SimplifierTopDown1e-0,SimplifierTopDown1e-1,SimplifierTopDown1e-2,SimplifierTopDown1e-4")

for model in "${models[@]}"
do
    # -repeats: runs per seed. -n_trials: number of different seeds
    python submit_jobs.py -repeats 1 -n_trials 30 -models "$model" -n_jobs 9 -data-dir ./data/lexicase_paper --local # --dryrun # 
done
