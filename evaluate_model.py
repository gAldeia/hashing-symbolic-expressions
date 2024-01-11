import argparse
import importlib
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
from pmlb import fetch_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error as mse
from  models.nsga2.deap_utils import get_complexity

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_data(dataset, random_state, data_dir):
    X, y = None, None
    if dataset in ['d_airfoil','d_concrete','d_enc','d_enh',
                   'd_housing','d_tower','d_uball5d','d_yacht']:
        
        data = pd.read_csv(f"{data_dir}/{dataset}.txt", sep=',')
        X, y = data.drop('label', axis=1).values, data['label'].values

    else: # We need to handle pmlb differently
        try:
            X, y = fetch_data(dataset, return_X_y=True, local_cache_dir=data_dir)
        except Exception as e:
            raise e
        
    if X is y and y is None:
        raise Exception("Unknown dataset")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state)
    
    return X_train, X_test, y_train, y_test 


def save_evolution(estimator, name, dataset, random_state, rdir, repeat):
    logbook = pd.DataFrame(columns=['gen', 'evals', 'n_simplifications', 'n_new_hashes'] + 
                            [f"{stat} {partition} {objective}"
                             for stat in ['avg', 'med', 'std', 'min', 'max']
                             for partition in ['train', 'val']
                             for objective in estimator.objectives])
    
    for item in estimator.logbook_:
        logbook.loc[item['gen']] = (
            item['gen'], item['evals'], item['n_simplifications'], item['n_new_hashes'],
            *item['avg'], *item['med'], *item['std'], *item['min'], *item['max']
        )

    logbook_filename = rdir + '_'.join([dataset, 
                                        name, 
                                        str(repeat), 
                                        str(random_state),
                                        "evolution"]) + '.csv'
        
    pd.DataFrame.from_dict({col: logbook[col] for col in logbook.columns}
    ).to_csv(logbook_filename, index=False)


def evaluate_model(
    estimator, name, dataset, random_state, rdir, repeat, data_dir='./'):
    """Evaluates estimator by training and predicting on the dataset."""

    X_train, X_test, y_train, y_test = read_data(dataset,random_state,data_dir)
    
    # set random states
    if hasattr(estimator, 'random_state'):
        estimator.random_state = random_state
    elif hasattr(estimator, 'seed'):
        estimator.seed = random_state
    
    print('algorithm:',algorithm.name,algorithm.reg)
    print('fitting to all data...')
    
    start_time = time.time()
    estimator.fit(X_train,y_train)
    end_time = time.time() - start_time
    
    if hasattr(estimator, 'logbook_') and estimator.logbook_ is not None:
        save_evolution(estimator,name,dataset,random_state,rdir,repeat)

    if "NSGAII" in estimator.__class__.__name__:
        model      = str(estimator.best_estimator_).replace("ARG", "x_")
        size       = len(estimator.best_estimator_)
        complexity = get_complexity(estimator.best_estimator_)
        depth      = estimator.best_estimator_.height
    else: # Assuming the only other model is the vanilla NSGA2
        model      = ""
        size       = 0
        complexity = size
        depth      = 0

    print(f'model {model}; size {size}, complexity {complexity}, depth {depth}')

    # get scores
    results = {}

    # Metadata
    results['model']        = name
    results['dataset']      = dataset
    results['RunID']        = repeat
    results['random_state'] = random_state
    results['time']         = end_time
    results['date']         = datetime.today().strftime('%m-%d-%Y %H:%M:%S')

    # metrics
    for metric, fn, (data_X, data_y) in [
        ('train_r2',  r2_score, (X_train, y_train)),
        ('test_r2',   r2_score, (X_test,  y_test )),
        ('train_mse', mse,      (X_train, y_train)),
        ('test_mse',  mse,      (X_test,  y_test )),
    ]:
        score = np.nan
        try:
            score = fn(estimator.predict(data_X), data_y)
        except ValueError:
            print(f"(Failed to calculate {metric} score for {name}")

        results[metric] = score

    # Model
    results['representation'] = model
    results['size']           = size
    results['complexity']     = complexity
    results['depth']          = depth

    print('results:',results)
    filename = (rdir + '_'.join([dataset, 
                            name, 
                            str(repeat),
                            str(random_state),
                            "result"]) + '.json')
    
    with open(filename, 'w') as out:
        json.dump(results, out, indent=4)


################################################################################
# main entry point
################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Evaluate a method on a dataset.", add_help=True)
    
    parser.add_argument('-ml', action='store', dest='ALG', default=None,
            type=str, 
            help='Name of estimator (with matching file in methods/)')
    
    parser.add_argument('-rdir', action='store', dest='RDIR',default=None,
            type=str, help='Name of save file')
    
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
            default=None, type=int, help='Seed / trial')
    
    parser.add_argument('-repeat', action='store', dest='REPEAT',
            default=1, type=int, help='repetition number')
    
    parser.add_argument('-dataset', action='store', dest='DATASET',
            default=None, type=str, help='endpoint name')
    
    parser.add_argument('-datadir', action='store', dest='DDIR',default='./',
            type=str, help='input data directory')

    args = parser.parse_args()
    print(args)
    
    # import algorithm 
    print('import from','models.'+args.ALG)
    algorithm = importlib.__import__('models.'+args.ALG,globals(),locals(),
                                     ['reg','name'])

    evaluate_model(algorithm.reg, algorithm.name, 
                   args.DATASET, args.RANDOM_STATE, args.RDIR, args.REPEAT,
                   data_dir = args.DDIR)
    