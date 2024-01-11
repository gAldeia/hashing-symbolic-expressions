# this file should unify the way we create the plots

from glob import glob
import pandas as pd
import seaborn as sns
import numpy as np

import sys
import os
import importlib.util

# pragmatically getting information about the number of generations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models._default_kwargs import kwargs as models_kwargs

pd.set_option('display.max_colwidth', None)
sns.set(style="ticks", palette='colorblind')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.0})

results_path = "../results"

step_size = 10 # How many generations to ignore between two points in the plots
skip_gens = 0 # How many initial generations to skip 
tot_gens  = models_kwargs['max_gen']

if not os.path.exists('../paper/figs'):
    os.makedirs('../paper/figs')

datasets = [
    'd_airfoil',
    'd_concrete',
    'd_enc',
    'd_enh',
    'd_housing',
    'd_tower',
    'd_uball5d',
    'd_yacht'
]
datasets_nice = [
    'Airfoil',
    'Concrete',
    "Energy Cooling",
    'Energy Heating',
    "Housing",
    "Tower",
    "UBall 5d",
    "Yacht"
]
dnames_to_nice = {k:v for k,v in zip(datasets, datasets_nice)}
dnames_to_ugly = {v:k for k,v in dnames_to_nice.items()}

objectives = ['error', 'size']

# Should be the folder name
# Comment models in/out in models and models_nice to include/exclude from results
model_filenames = [
    file for file in glob('../models/*.py') if not file.split('/')[-1].startswith('_')
]

model_folder = [
    filename.split('../models/')[1].split('.py')[0]
    for filename in model_filenames
]

# Extracting the name from the files
def get_name_value(script_path):
    with open(script_path, 'r') as file:
       lines = file.readlines()
       for line in lines:
        if line.strip().startswith('name ='):
            return line.split('=')[1].strip().replace("'", "\"").replace("\"", "")

model_nice = [
    get_name_value(filename)
    for filename in model_filenames
]

print(model_filenames, model_nice)

markers = ('^','o', 's', 'p', 'P', 'h', 'D', 'P', 'X', 'v', '<', '>','*')
order = model_nice #sorted(model_nice)

marker_choice = { model: marker for (model, marker) in zip(model_nice, markers) }

# how we sample the generations
gens = range(tot_gens)                         # all generations (slower)
# gens = range(skip_gens, tot_gens, step_size)   # skipping generations
# gens = [int(np.floor(g)) for g in np.logspace( # skipping with log scale
#                                        np.log10(0.99+skip_gens),
#                                        np.log10(tot_gens),
#                                        num=tot_gens//step_size)]

# Loading overall results
results = []
for dataset in datasets: # this will iterate over keys
    for model in model_folder:
        for file in glob(f"{results_path}/{dataset}/{model}/*.json"):
            df = pd.read_json(file, typ='series')

            # Filtering columns if needed
            indxs = [indx for indx in df.index if indx not in [None]]

            results.append( df[indxs] )

results_df = pd.DataFrame(data=results, columns=indxs)

# Beautifying it
results_df['dataset'] = results_df['dataset'].apply(lambda t: dnames_to_nice[t])

print(results_df.shape)
print(results_df['model'].unique())
print(results_df['dataset'].unique())

boxplot_kwargs = { # Standarized boxplot style
    'kind'        : "box",
    'dodge'       : False,
    'showfliers'  : False,                 
    'flierprops'  : {"marker": "x"},
    'notch'       : True,
    'orient'      : "v",
    'estimator'   : np.median, 
    'boxprops'    : {"facecolor": 'white'},
    'medianprops' : {"color": "k", "linewidth": 3},
    'showcaps'    : False,
}