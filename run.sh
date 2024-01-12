#!/bin/bash

models=("SimplifierBottomUp,SimplifierTopDown,Vanilla")

for model in "${models[@]}"
do
    # -repeats: runs per seed. -n_trials: number of different seeds
    python submit_jobs.py -repeats 1 -n_trials 30 -models "$model" -n_jobs 12 -data-dir ./data/lexicase_paper --local # --dryrun # 
done
