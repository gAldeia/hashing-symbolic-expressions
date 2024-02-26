# Implement a hash-based "simplify" function to be imported by nsga2_deap
# Created by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

# https://github.com/loretoparisi/lshash
import lshashpy3 as lshash
import bisect
import warnings
import numpy as np
from deap import gp

from .deap_utils import get_complexity


class HashSimplifier:
    def __init__(self, Individual, Fitness, toolbox,
                 hash_len=256, tolerance=1e-20):
        self.Individual = Individual
        self.Fitness = Fitness # not used
        self.toolbox = toolbox
        self.hash_len = hash_len # number of bits used for the hash
        self.tolerance = tolerance # the distance tolerance to consider two individual the same
        self.initialized = False

        self.num_hashtables = 5
        self.distance_func = "l1norm"
        self.key = lambda ind: len(ind)
        #self.key = lambda ind: get_complexity(ind)


    def _is_equal(self, ind1, ind2):
        if len(ind1) != len(ind2):
            return False
        
        for node1, node2 in zip(ind1, ind2):
            # If it is a constant, we dont care about the value
            if  isinstance(type(node1), gp.MetaEphemeral) \
            and isinstance(type(node2), gp.MetaEphemeral) :
                continue
            elif node1.name == node2.name:
                continue
            else:
                return False

        return True
    

    def _is_in(self, ind, pop_index):
        # print("checking for similarity")
        for idx, tree in enumerate(self.pop_hash[pop_index]):
            # Finding indexes where the size matches
            if len(ind) > len(tree):
                continue
            elif len(ind) < len(tree):
                break
            elif self._is_equal(ind, tree):
                return idx
        return -1
    

    def _predict_hash(self, ind, X, y):
        # auxiliary function to compile an individual and evaluate it on the data

        expr = self.toolbox.compile(expr=ind)
        pred = np.array([expr(*x) for x in X])

        # constant predictions, should be mapped into dummy_ind hash
        if np.std(pred) <= 1e-10:
           # print('CONST SEMANTIC')
           return np.ones_like(pred)
        
        return pred # np.std(pred) * ( pred + np.mean(pred) )
    

    def initialize(self, pset, X, y):
        # Use the first individual to extract info to initialize the table

        # Creating our locality sensitive hashing table
        self.lsh = lshash.LSHash(self.hash_len, len(y), num_hashtables=self.num_hashtables)
        
        # This is the memoization part
        # Adding the first value into the memoization
        self.pop_hash = {}
        
        # inserting each terminal into the hash table
        for i in range(X.shape[1]):
            feature_ind = self.Individual([pset.mapping[f'ARG{i}']])

            # Getting the semantics for the individual
            h = self._predict_hash(feature_ind, X, y)

            # Setting the first index
            self.lsh.index(h, extra_data=str(i+1))

            self.pop_hash[str(i+1)] = [feature_ind]

        dummy_ind = self.Individual([pset.mapping['rand100']()])
        dummy_ind[0].value = 1.0 # setting the value to be 1, same as constant nodes would have

        h = self._predict_hash(dummy_ind, X, y)

        self.lsh.index(h, extra_data="0")

        # Inserting const last --- so it overrides a constant feature
        self.pop_hash["0"] = [dummy_ind]
        
        # Counters for statistics
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        if len(self.pop_hash.keys()) != X.shape[1]+1:
            # There is the case where we have one constant feature
            # (few samples, bad train/test split, bad data...)
            message  = ("It was not possible to create one hash for each terminal " 
                        "(which means that there is a hash colision). Try to "
                        "increase hash size.")

            warnings.warn(message)
            #raise Exception(message)

        self.toolbox.register("get_n_simplifications", lambda: self.n_simplifications)
        self.toolbox.register("get_n_new_hashes", lambda: self.n_new_hashes)

        self.initialized = True
        
        return self


    def simplify_pop_bottom_up(self, pop, X, y, replace_pop=True):
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        new_pop = []
        refit = []
        for idx in range(len(pop)):
            ind = self.toolbox.clone(pop[idx])
            # print(f'simplifying ind idx{idx} {ind}')
            original_str = str(ind)
            
            refit_ind = False
            indexes = list(enumerate(ind))[::-1]
            # print(f' - list of indexes {indexes}')
            for idx_node, node in indexes:
                if isinstance(node, gp.Terminal):
                    # print(f'     - skipping')
                    continue
                
                # print(f'   - {idx_node}, {node.name}')
                ind_subtree = ind[ind.searchSubtree(idx_node)]
                # print(f'   - subtree {ind_subtree}')

                ind_subtree = self.Individual(self.toolbox.clone(ind_subtree)[:])
                # print(f'     - cast into ind {ind_subtree}')

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)
                # print(f'     - semantics {h[:3]}')

                if not np.all(np.isfinite(h)):
                    # print(f'     - bad semantics')
                    continue

                pop_index, d = None, np.inf
                try: 
                    # Querying for the closest result
                    res = self.lsh.query(h, num_results=self.num_hashtables, distance_func=self.distance_func)
                    
                    # print(f'       - query {res}')
                    data, d      = res[0]
                    v, pop_index = data

                    # print(f'       - closest hash {pop_index} with dist {d} and semantics {v[:3]}')
                    # print(f'       - ind is {self.pop_hash[pop_index][0]}')
                    for indf in  self.pop_hash[pop_index][1:]:
                        # print(f'       - (next ) {indf}')
                        pass
                except IndexError:
                    # print(f'       - indexing it (failed to find any hash)')
                    pop_index = str(len(self.pop_hash))

                    pass

                # print(f'       - tolerance check {d} <= {self.tolerance }')    
                if d <= self.tolerance : # they are similar
                    # print(f'         - success, trying to simplify')
                
                    if len(self.pop_hash[pop_index][0]) < len(ind_subtree):
                        
                        # print(f'         - can be simplified cause len {self.key(self.pop_hash[pop_index][0])} of {self.pop_hash[pop_index][0]} < len {self.key(ind_subtree)} of {ind_subtree}')
                        
                        # Several simplifications can happen in a single individual
                        self.n_simplifications += 1

                        # We need to wrap here to allow the loop to continue
                        # (as it uses some of PrimitiveTree methods)
                        ind = self.Individual( ind[:idx_node] +\
                                               self.toolbox.clone(self.pop_hash[pop_index][0])[:] +\
                                               ind[idx_node+len(ind_subtree):] )

                        refit_ind = True
                        # print(f'         - result {ind}')
                    else:
                        # print(f'         - CANNOT be simplified cause len {self.key(self.pop_hash[closest_hash][0])} of {self.pop_hash[closest_hash][0]} >= len {self.key(ind_subtree)} of {ind_subtree}')
                        pass

                    # avoid repeated hashes (this is good for statistics)
                    if self._is_in(ind_subtree, pop_index) == -1:
                    
                        self.n_new_hashes += 1
                        bisect.insort(
                            self.pop_hash[pop_index], ind_subtree,
                            key=lambda ind: self.key(ind))
                    
                if pop_index not in self.pop_hash:
                    # print(f'         - created new hash')
                    self.pop_hash[pop_index] = [ind_subtree]
                    self.n_new_hashes += 1
                    self.lsh.index( h, extra_data = pop_index )

                # else: # Stop inserting thrash into the dict
                #     bisect.insort(
                #         self.pop_hash[pop_index], ind_subtree,
                #         key=lambda ind: self.key(ind))

            assert original_str == str(pop[idx])

            new_pop.append(ind)
            if refit_ind:
                refit.append(idx)

        if replace_pop:
            return new_pop, refit
        
        self.n_simplifications=0# ignoring simplifications, as we are returning original pop
        return pop, []
    

    def simplify_pop_top_down(self, pop, X, y, replace_pop=True):
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        new_pop = []
        refit = []
        for idx in range(len(pop)):
            ind = self.toolbox.clone(pop[idx])

            original_str = str(ind)
            refit_ind = False

            indexes = range(0, len(ind))
            while len(indexes)>1:
                # print( indexes[0], len(indexes), len(ind))
                idx_node = indexes[0]
                indexes  = indexes[1:]

                ind_subtree = ind[ind.searchSubtree(idx_node)]

                if len(ind_subtree) <=1:
                    continue

                ind_subtree = self.Individual(self.toolbox.clone(ind_subtree)[:])

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)

                if not np.all(np.isfinite(h)):
                    continue

                pop_index, d = None, np.inf
                try:
                    # Querying for the closest result
                    res = self.lsh.query(h, num_results=self.num_hashtables, distance_func=self.distance_func)
                    
                    data, d      = res[0]
                    v, pop_index = data
                except IndexError:
                    pop_index = str(len(self.pop_hash))

                if d <= self.tolerance : # they are similar
                    if len(self.pop_hash[pop_index][0]) < len(ind_subtree):
                        self.n_simplifications += 1
                        
                        ind = self.Individual(ind[:idx_node] + \
                                self.toolbox.clone(self.pop_hash[pop_index][0])[:] + \
                                ind[idx_node+len(ind_subtree):])

                        refit_ind = True

                        indexes = range(idx_node+1, len(ind)) #[1:]

                    # avoid repeated hashes (this is good for statistics)
                    if self._is_in(ind_subtree, pop_index) == -1:
                        
                        self.n_new_hashes += 1
                        bisect.insort(
                            self.pop_hash[pop_index], ind_subtree,
                            key=lambda ind: self.key(ind))
                    
                if pop_index not in self.pop_hash:
                    # print(f'         - created new hash')
                    self.pop_hash[pop_index] = [ind_subtree]
                    self.n_new_hashes += 1
                    self.lsh.index( h, extra_data = pop_index )

            assert original_str == str(pop[idx])
            
            new_pop.append(ind)
            if refit_ind:
                refit.append(idx)

        if replace_pop:
            return new_pop, refit
        
        self.n_simplifications=0# ignoring simplifications, as we are returning original pop
        return pop, []