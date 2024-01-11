# Baseline estimator
# Created by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

import operator
import warnings

import numpy as np

from .deap_utils import PTC2_deap, node_functions, ERC100, get_complexity
from .nsga2_deap import nsga2_deap
from .optimizer import optimize_individual
from .hash_simplifier import HashSimplifier

from deap import base, creator, tools,gp

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError


class NSGAIIEstimator(BaseEstimator):
    def __init__(
        self, 
        pop_size=100,
        max_gen=100,
        max_depth=3,
        max_size=20,
        objectives=['error', 'size'],
        cx_prob=1/7,
        functions: list[str] = [],
        initialization = 'uniform',
        pick_criteria = 'MCDM', 
        validation_size: float = 0.0, 
        simplify=True,
        simplification_method="bottom_up",
        verbosity=0,
        mode='regression',
        **kwargs
    ):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.verbosity=verbosity
        self.max_depth=max_depth
        self.max_size=max_size
        self.cx_prob=cx_prob
        self.functions=functions
        self.initialization=initialization
        self.pick_criteria=pick_criteria
        self.validation_size=validation_size
        self.simplify = simplify
        self.simplification_method=simplification_method
        self.objectives = objectives
        self.mode=mode

        self._is_fitted = False

        self.random = np.random.default_rng()


    def _fitness_validation(self, ind, X, y):
        # Fitness without fitting the expression, used with validation data

        ind_objectives = {
            "error"     : self._error(ind, X, y),
            "size"      : len(ind),
            "complexity": get_complexity(ind)
        }

        return [ ind_objectives[obj] for obj in self.objectives ]


    def _fitness_function(self, ind, X, y):
        # fit the expression, then evaluate.
        ind = optimize_individual(self.toolbox_, ind, X, y)

        return self._fitness_validation(ind, X, y)
    

    def _setup_toolbox(self, X_train, y_train, X_val, y_val):
        pset = gp.PrimitiveSet("MAIN", arity=X_train.shape[1]) 

        # TODO: have  a dict  and iterate through it to add the primitives to the  set
        if self.functions == []:
            self.functions = [
                    'div', 'add', 'sub', 'mul',
                    'add3', 'add4', 'mul3', 'mul4',
                    'maximum', 'minimum',
                    'sin', 'cos', 'tan',
                    'sqrtabs', 'log1p', 'exp', 'square', 'abs'
            ]

        for f in self.functions:
            func, arity = node_functions[f]
            pset.addPrimitive(func, arity)

        # This is required to have optimizable paremeters
        pset.addEphemeralConstant("rand100", ERC100)

        # DEAP Toolbox
        toolbox = base.Toolbox()

        #  TODO: instead of using creator to create those classes, implement them!

        # Cleaning possible previous classes that are model-dependent
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Individual creation
        creator.create("FitnessMulti", base.Fitness, weights=self.weights)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)  

        # Initial population 
        toolbox.register("max_size_sampler",
                         lambda: 3+self.random.choice(self.max_size) if self.initialization=='uniform' else self.max_size)
        toolbox.register("expr", PTC2_deap, pset=pset, min_=1, max_=self.max_depth, max_size_sampler=toolbox.max_size_sampler)
        
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Fitness Evaluation
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self._fitness_function, X=X_train, y=y_train)
        toolbox.register("evaluateValidation", self._fitness_validation, X=X_val, y=y_val)

        # Variation operators: mutation 
        toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))	
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=self.max_size))

        # Variation operators: crossover
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=self.max_size))

        # Selection and survival steps 
        toolbox.register("get_objectives", lambda: ['error', 'size'])
        toolbox.register("select", tools.selTournamentDCD)
        toolbox.register("survive", tools.selNSGA2)

        # Optimize individual
        simplifier = HashSimplifier(creator.Individual, creator.FitnessMulti, toolbox)
        simplifier.initialize(pset, X_train, y_train)

        toolbox.register("get_n_simplifications", lambda: simplifier.n_simplifications)
        toolbox.register("get_n_new_hashes", lambda: simplifier.n_new_hashes)

        if self.simplification_method=="bottom_up":
            toolbox.register("simplify_pop", simplifier.simplify_pop_bottom_up)
        elif self.simplification_method=="top_down":
            toolbox.register("simplify_pop", simplifier.simplify_pop_top_down)
        else:
            raise Exception("Unknown simplification method")
        
        return toolbox


    def fit(self, X, y):
        obj_weight = {
            "error"      : +1.0 if self.mode=="classification" else -1.0,
            "size"       : -1.0,
            "complexity" : -1.0
        }
        self.weights = [obj_weight[w] for w in self.objectives]

        X_train, X_val = X, X
        y_train, y_val = y, y
        
        if self.validation_size>0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_size)
            
        self.toolbox_ = self._setup_toolbox(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        with warnings.catch_warnings():
            # numeric errors, overflow, invalid value for specific math functions
            warnings.simplefilter('ignore', category=RuntimeWarning)

            archive, logbook = nsga2_deap(
                self.toolbox_, self.max_gen, self.pop_size, self.cx_prob,
                self.verbosity, self.random, self.simplify, X_train, y_train)

        self.archive_ = archive
        self.logbook_ = logbook

        # ----------------------------------------------------------------------
        final_ind_idx = 0

        # Each individual is a point in the Multi-Objective space. We multiply
        # the fitness by the weights so greater numbers are always better
        points = np.array([self.toolbox_.evaluateValidation(ind) for ind in self.archive_])
        points = points*np.array(self.weights)

        # Using the multi-criteria decision making on:
        # - test data if pick_criteria is MCDM
        if self.pick_criteria=="MCDM":
            # Selecting the best estimator using training data
            # (train data==val data if validation_size is set to 0.0)
            # and multi-criteria decision making

            # Normalizing
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            points = (points - min_vals) / (max_vals - min_vals)
            points = np.nan_to_num(points, nan=0)
            
            # Reference should be best value each obj. can have (after normalization)
            reference = [1 for _ in range(len(self.weights))]

            # closest to the reference
            final_ind_idx = np.argmin( np.linalg.norm(points - reference, axis=1) )
        else: # Best in obj.1 (loss) in validation data (or training data)
            final_ind_idx = max(
                range(len(points)),
                key=lambda index: (points[index][0], points[index][1]) )

        self.best_estimator_ = self.archive_[final_ind_idx]
        # ----------------------------------------------------------------------

        self._expr = self.toolbox_.compile(expr=self.best_estimator_)
        
        if self.verbosity > 0:
            print(f'best model {str(self.best_estimator_).replace("ARG", "x_")}' +
                  f' with size {len(self.best_estimator_)}, ' +
                  f' depth {self.best_estimator_.height}, '   +
                  f' and fitness {self.best_estimator_.fitness}'  )

        self._is_fitted = True
        
        return self
    
    def get_params(self, deep=True):
        return {k:v for k,v in self.__dict__.items() if  not k.endswith('_')
                                                     and not k.startswith('_')}


class NSGAIIRegressor(NSGAIIEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(mode='regression', **kwargs)


    def _error(self, ind, X, y):
        expr = self.toolbox_.compile(expr=ind)
        
        pred = np.array([expr(*x) for x in X])
        MSE  = np.mean( (y-pred)**2 )

        if not np.isfinite(MSE):
            MSE = np.inf

        return MSE
    

    def predict(self, X):
        if not self._is_fitted:
            raise NotFittedError(
                "The expression was simplified and has not refitted.")
        
        pred = np.array([self._expr(*x) for x in X])

        return pred