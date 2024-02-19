# implements variation operators with MABs

import lshashpy3 as lshash
import bisect
import warnings
import numpy as np

from functools import partial
import operator
from .deap_utils import get_complexity
from deap import base, creator, tools, gp

from .variator_with_lsh import HashVariator
from .MAB.Listener          import Listener
from .MAB.UCB1Learner       import UCB1Learner
from .MAB.ContextualWrapper import ContextSpace, ContextualWrapper

# TODO: import variator with lsh and use it if smart_variation is set to true
class Variator:
    def __init__(self, Individual, Fitness, toolbox, rnd_generator,
                 max_depth, max_size,
                 use_mab=False, use_context=False, smart_variation=False):
        
        self.Individual = Individual
        self.Fitness    = Fitness
        self.toolbox    = toolbox
        self.max_depth  = max_depth
        self.max_size   = max_size

        self.rnd_generator = rnd_generator

        self.use_mab         = use_mab
        self.use_context     = use_context
        self.smart_variation = smart_variation
        
        self.initialized = False
        
        self.key = lambda ind: len(ind)
        #self.key = lambda ind: get_complexity(ind)

        # Hardcoding this because it makes the algorithm slower, and is for development purposes
        self.log_file = False


    def _predict_hash(self, ind, X, y):
        # auxiliary function to compile an individual and evaluate it on the data

        expr = self.toolbox.compile(expr=ind)
        pred = np.array([expr(*x) for x in X])

        stardized_pred = (pred - np.mean(pred)) + 1e-8
        
        return stardized_pred
    

    def initialize(self, pset, X, y, delete_at):
        
        if self.log_file is not None:
            self.log = {c:[] for c in
                        ['gen', 'variation', 'euclid dist prediction'] + \
                        [f"delta {obj}"
                         for obj in self.toolbox.get_objectives()]}

        # Variation operators: mutations (4 types, and a main function to wrap it).
        # mutations from deap returns a list with 1 individual
        if self.smart_variation:
            self.variator_ = HashVariator(
                self.Individual, self.toolbox, self.rnd_generator
            ).init(pset, X, y)

            self.mutations = {
                "lsh_mutate" : self.variator_.mutate,
                "subtree"    : self.variator_.subtree,
            }

            self.CXPB      = 1/5
            self.mut_probs = { "lsh_mutate" : 1/2, 'subtree' : 1/2}
            
            # We need something with fixed order of the mutations
            self.arm_labels = ['lsh_mutate', 'subtree', 'cx']
        else:
            self.mutations = {
                "point"   : partial(gp.mutNodeReplacement, pset=pset),
                "delete"  : gp.mutShrink,
                "subtree" : partial(gp.mutUniform, pset=pset, expr=self.toolbox.expr),
                "insert"  : partial(gp.mutInsert, pset=pset)
            }

            # Initializing probabilities with uniform distribution
            self.CXPB      = 1/(1 + len(self.mutations))
            self.mut_probs = { "point"   : 1/len(self.mutations),
                               "delete"  : 1/len(self.mutations),
                               "subtree" : 1/len(self.mutations),
                               "insert"  : 1/len(self.mutations)}

            # We need something with fixed order of the mutations
            self.arm_labels = ['point', 'delete', 'subtree', 'insert', 'cx']
        
        if self.use_mab:
            if self.use_context:
                self.lsh= lshash.LSHash(256, len(y), num_hashtables=5)

                self.const_hash = 1 # Skip zero
                self.lsh.index(np.zeros_like(y), extra_data=self.const_hash)
                
                hash_space = [self.const_hash]
                for i in range(X.shape[1]):
                    self.lsh.index(X[:, i], extra_data=i+2)
                    hash_space.append(i+2)

                context_spaces = ContextSpace(
                    np.linspace(0, 200, num=10_000), # error
                    np.linspace(0, 128, num=128),    # size
                    hash_space                       # hash (*should be sorted*)
                )

                self.mab = ContextualWrapper(
                    n_obj=2, n_arms=len(self.arm_labels), 
                    arm_labels=self.arm_labels, rnd_generator=self.rnd_generator,
                    context_keys=['error', 'size', 'hash'], context_space=context_spaces,
                    delete_at=delete_at, Learner=UCB1Learner, Learner_kwargs={},
                )
            else:
                self.mab = UCB1Learner(
                    n_obj=2, n_arms=len(self.arm_labels), arm_labels=self.arm_labels)
        else:
            self.mab = Listener(
                n_obj=2, n_arms=len(self.arm_labels), arm_labels=self.arm_labels)

        def mutate(ind, mut):
            return self.mutations[mut](ind)
        
        self.toolbox.register("mutate", mutate)
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))	
        self.toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=self.max_size))
        
        # Variation operators: crossover
        if self.smart_variation and False:
            self.toolbox.register("crossover", self.variator_.cross)
        else:        
            def crossover(ind1, ind2):
                return gp.cxOnePoint(ind1, ind2)
            self.toolbox.register("crossover", crossover)
        self.toolbox.decorate("crossover", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        self.toolbox.decorate("crossover", gp.staticLimit(key=len, max_value=self.max_size))

        self.toolbox.register("vary_pop", self.vary_pop)
        
        return self
    
    
    def warm_up(self, pop):
        if self.smart_variation:
            for ind in pop:
                self.variator_._memoize_and_find_spots(ind, constants_not_allowed=True)
        return 
    

    def vary_pop(self, parents, gen, X, y):
        offspring = []
        for ind1, ind2 in zip(parents[::2], parents[1::2]):
            for ind in [ind1, ind2]:
                pop_index = None
                if self.use_mab and self.use_context:
                    try:
                        h = self._predict_hash(ind, X, y)
                        res = self.lsh.query(h, num_results=3, distance_func="l1norm")
                    
                        # print(res)
                        h, pop_index = res[0][0]
                    except IndexError:
                        pop_index = self.const_hash
            
                ctx = { 'gen'   : gen,
                        'size'  : len(ind),
                        'error' : ind.fitness.values[0],
                        'hash'  : pop_index}
                
                if self.use_mab:
                    variation = self.mab.choose_arm(ctx)
                else:
                    variation = 'cx'
                    if self.rnd_generator.random() > self.CXPB:
                        variation = self.rnd_generator.choice(
                            list(self.mut_probs.keys()),
                            p=list(self.mut_probs.values())
                        )
                
                off = self.toolbox.clone(ind1)
                if  variation == 'cx':
                    off2 = self.toolbox.clone(ind2)

                    off, _ = self.toolbox.crossover(off, off2)
                else:
                    # workaround for DEAP returning one random argument of
                    # mutate when it exceeds static limits of size or depth
                    xmen, = self.toolbox.mutate(off, variation)
                    if (xmen != variation):
                        off = xmen
                        
                # Refit the individual after variation
                off.fitness.values = self.toolbox.evaluate(off)
                delta_costs = np.subtract(ind.fitness.values, off.fitness.values)

                if self.log_file is not None:
                    # Finding the spot where variation occured
                    variation_index = 0
                    for index, (node1, node2) in enumerate(zip(ind[:], off[:])):
                        if  isinstance(type(node1), gp.MetaEphemeral) \
                        and isinstance(type(node2), gp.MetaEphemeral) :
                            continue
                        elif node1.name == node2.name:
                            continue
                        else:
                            variation_index = index
                            break
                        
                    original_vector = self._predict_hash(ind, X, y)
                    new_vector      = self._predict_hash(off, X, y)

                    euclid_dist = np.linalg.norm(original_vector-new_vector)

                    self.log['gen'].append(gen)
                    self.log['variation'].append(variation)
                    self.log['euclid dist prediction'].append(euclid_dist)

                    for obj, val in zip(self.toolbox.get_objectives(),
                                        delta_costs*ind.fitness.weights):
                        self.log[f"delta {obj}"].append(val) 

                # In case our MAB is the listener, it will only log
                self.mab.update(
                    arm         = variation,
                    delta_costs = delta_costs,
                    context     = ctx
                )

                offspring.extend([off])

        return offspring
