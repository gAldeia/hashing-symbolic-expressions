#!/bin/bash

models=()

#models+=("Vanilla,SimplifierBottomUp,SmartVariation,SmartVariationSimplify,SmartVariationSimplifyMAB,SmartVariationMAB") # Vanilla,
# models+=("Vanilla,SmartVariationMAB") # Vanilla,
#models+=("SmartVariationMAB") # Vanilla,
# models+=("SimplifierBottomUpCMAB") # Vanilla,SimplifierBottomUp,SimplifierBottomUpMAB,
# models+=("SimplifierBottomUpOnlyLast,SimplifierTopDownOnlyLast")

models+=("Vanilla")

# models+=("SimplifierBottomUp")
# models+=("SimplifierTopDown")

# models+=("SimplifierBottomUpOnlyLast")
# models+=("SimplifierTopDownOnlyLast")

models+=("SimplifierTopDown1e-1,SimplifierTopDown1e-3,SimplifierTopDown1e-5")
models+=("SimplifierBottomUp1e-1,SimplifierBottomUp1e-3,SimplifierBottomUp1e-5")
models+=("SimplifierTopDownOnlyLast1e-1,SimplifierTopDownOnlyLast1e-3,SimplifierTopDownOnlyLast1e-5")
models+=("SimplifierBottomUpOnlyLast1e-1,SimplifierBottomUpOnlyLast1e-3,SimplifierBottomUpOnlyLast1e-5")

for model in "${models[@]}"
do
    # -repeats: runs per seed. -n_trials: number of different seeds
    
    # python submit_jobs.py -repeats 1 -n_trials 10 -models "$model" -n_jobs 1 -data-dir ./data/lexicase_paper --slurm -time 6:00:00 -m 3000
    python submit_jobs.py -repeats 1 -n_trials 10 -models "$model" -n_jobs 60 -data-dir ./data/lexicase_paper --local
done
