#!/bin/bash

models=()

#models+=("Vanilla,SimplifierBottomUp,SmartVariation,SmartVariationSimplify,SmartVariationSimplifyMAB,SmartVariationMAB") # Vanilla,
models+=("Vanilla,SmartVariationMAB") # Vanilla,
#models+=("SmartVariationMAB") # Vanilla,
models+=("SimplifierBottomUpCMAB") # Vanilla,SimplifierBottomUp,SimplifierBottomUpMAB,
# models+=("SimplifierBottomUpOnlyLast,SimplifierTopDownOnlyLast")

# models+=("Vanilla")

# models+=("SimplifierBottomUp")
# models+=("SimplifierTopDown")

# models+=("SimplifierBottomUpOnlyLast")
# models+=("SimplifierTopDownOnlyLast")

# models+=("SimplifierTopDown1e-0,SimplifierTopDown1e-1")
# models+=("SimplifierTopDown1e-2,SimplifierTopDown1e-4")
# models+=("SimplifierTopDown1e-6,SimplifierTopDown1e-10")

# models+=("SimplifierBottomUp1e-0,SimplifierBottomUp1e-1")
# models+=("SimplifierBottomUp1e-2,SimplifierBottomUp1e-4")
# models+=("SimplifierBottomUp1e-6,SimplifierBottomUp1e-10")

for model in "${models[@]}"
do
    # -repeats: runs per seed. -n_trials: number of different seeds
    
    # python submit_jobs.py -repeats 1 -n_trials 10 -models "$model" -n_jobs 1 -data-dir ./data/lexicase_paper --slurm -time 6:00:00 -m 3000
    python submit_jobs.py -repeats 1 -n_trials 5 -models "$model" -n_jobs 5 -data-dir ./data/lexicase_paper --local
done
