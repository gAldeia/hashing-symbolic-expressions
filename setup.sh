#!/bin/bash
source activate base

# set our conda environment
if conda info --envs | grep -q hashing-experiments;
    then echo "hashing-experiments env already exists";
    else conda env create -f environment.yml;
fi

conda activate hashing-experiments