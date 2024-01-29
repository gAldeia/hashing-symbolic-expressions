# Implement a hash-based "simplify" function to be imported by nsga2_deap
# Created by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

# https://github.com/loretoparisi/lshash
import lshashpy3 as lshash
import bisect
import warnings
import numpy as np

from .deap_utils import get_complexity

class HashSimplifier:
    def __init__(self, Individual, Fitness, toolbox,
                 hash_len=256, tolerance=1e-20):
        self.Individual = Individual
        self.Fitness = Fitness
        self.toolbox = toolbox
        self.hash_len=hash_len # number of bits used for the hash
        self.tolerance=tolerance # the distance tolerance to consider two individual the same
        self.initialized = False
    
        self.key = lambda ind: len(ind)
        #self.key = lambda ind: get_complexity(ind)


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
        self.lsh = lshash.LSHash(self.hash_len, len(y))
        
        # This is the memoization part
        # Adding the first value into the memoization
        self.pop_hash = {}
        
        # inserting each terminal into the hash table
        for i in range(X.shape[1]):
            feature_ind = self.Individual([pset.mapping[f'ARG{i}']])

            # Getting the semantics for the individual
            h = self._predict_hash(feature_ind, X, y)

            # Setting the first index
            self.lsh.index(h)

            # getting the hash representation for the prediction vector 'h'
            binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)

            # If two features ends with same hash, one is going to be overriten
            self.pop_hash[binary_hash] = [feature_ind]

        dummy_ind = self.Individual([pset.mapping['rand100']()])
        dummy_ind[0].value = 0.0 # setting the value to be zero, same as constant nodes would have

        h = self._predict_hash(dummy_ind, X, y)

        self.lsh.index(h)

        binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)

        # Inserting const last --- so it overrides a constant feature
        self.pop_hash[binary_hash] = [dummy_ind]
        
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

        self.initialized = True


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
                # print(f'   - {idx_node}, {node.name}')
                ind_subtree = ind[ind.searchSubtree(idx_node)]
                # print(f'   - subtree {ind_subtree}')

                if len(ind_subtree) <=1:
                    # print(f'     - skipping')
                    continue

                ind_subtree = self.Individual(self.toolbox.clone(ind_subtree)[:])
                # print(f'     - cast into ind {ind_subtree}')

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)
                # print(f'     - semantics {h[:3]}')

                if np.any(np.bitwise_not(np.isfinite(h))):
                    # print(f'     - bad semantics')
                    continue

                # Hash for the individual
                binary_hash  = self.lsh._hash(self.lsh.uniform_planes[0], h)
                
                closest_hash = None
                try: 
                    # Querying for the closest result
                    res = self.lsh.query(h, num_results=3, distance_func="euclidean")
                    
                    # print(f'       - query {res}')
                    ((v, extra), d) = res[0]
                
                    closest_hash = self.lsh._hash(self.lsh.uniform_planes[0], v)

                    # print(f'     - hash {binary_hash}')
                    # print(f'       - closest hash {closest_hash} with dist {d} and semantics {v[:3]}')
                    # print(f'       - ind is {self.pop_hash[closest_hash][0]}')
                    for indf in  self.pop_hash[closest_hash][1:]:
                        # print(f'       - (next ) {indf}')
                        pass
                except IndexError:
                    # print(f'       - indexing it (failed to find any hash)')

                    if binary_hash not in self.pop_hash:
                        # print(f'         - created new hash')
                        self.pop_hash[binary_hash] = [ind_subtree]
                        self.n_new_hashes += 1
                        self.lsh.index( h )
                    # else: # Stop inserting thrash into the dict
                    #     bisect.insort(
                    #         self.pop_hash[binary_hash], ind_subtree,
                    #         key=lambda ind: self.key(ind))

                    continue

                # print(f'       - tolerance check {d} <= {self.tolerance }')    
                if d <= self.tolerance : # they are similar
                    # print(f'         - success, trying to simplify')
                
                    if len(self.pop_hash[closest_hash][0]) < len(ind_subtree):
                        
                        # print(f'         - can be simplified cause len {self.key(self.pop_hash[closest_hash][0])} of {self.pop_hash[closest_hash][0]} < len {self.key(ind_subtree)} of {ind_subtree}')
                        
                        # Several simplifications can happen in a single individual
                        self.n_simplifications += 1

                        # We need to wrap here to allow the loop to continue
                        # (as it uses some of PrimitiveTree methods)
                        ind = self.Individual( ind[:idx_node] +\
                                               self.toolbox.clone(self.pop_hash[closest_hash][0])[:] +\
                                               ind[idx_node+len(ind_subtree):] )

                        refit_ind = True
                        # print(f'         - result {ind}')
                    else:
                        # print(f'         - CANNOT be simplified cause len {self.key(self.pop_hash[closest_hash][0])} of {self.pop_hash[closest_hash][0]} >= len {self.key(ind_subtree)} of {ind_subtree}')
                        pass

                    # avoid repeated hashes (this is good for statistics)
                    if len([i for i in self.pop_hash[closest_hash]
                            if str(i) == str(ind_subtree)]) == 0:
                        
                        self.n_new_hashes += 1
                        bisect.insort(
                            self.pop_hash[closest_hash], ind_subtree,
                            key=lambda ind: self.key(ind))
                    
                else: # Making it a new semantics item
                    # print(f'         - not simplifiable, will insert it')
                    if binary_hash not in self.pop_hash:
                        # print(f'           - created new hash')
                        self.pop_hash[binary_hash] = [ind_subtree]
                        self.n_new_hashes += 1
                        self.lsh.index( h )
                    else:
                        # print(f'           - similar hash, family of {self.pop_hash[binary_hash][0]}. skipping')
                        # bisect.insort(
                        #     self.pop_hash[binary_hash], ind_subtree,
                        #     key=lambda ind: len(ind))
                        continue

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

                if np.any(np.bitwise_not(np.isfinite(h))):
                    continue

                # Hash for the individual
                binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
                
                closest_hash = None
                try:
                    # Querying for the closest result
                    res = self.lsh.query(h, num_results=3, distance_func="euclidean")
                    
                    ((v, extra), d) = res[0]
                    closest_hash = self.lsh._hash(self.lsh.uniform_planes[0], v)
                except IndexError:
                    if binary_hash not in self.pop_hash:
                        self.pop_hash[binary_hash] = [ind_subtree]
                        self.n_new_hashes += 1
                        self.lsh.index( h )
                    continue

                if d <= self.tolerance : # they are similar
                    if len(self.pop_hash[binary_hash][0]) < len(ind_subtree):
                        self.n_simplifications += 1
                        
                        ind = self.Individual(ind[:idx_node] + \
                                self.toolbox.clone(self.pop_hash[binary_hash][0])[:] + \
                                ind[idx_node+len(ind_subtree):])

                        refit_ind = True

                        indexes = range(idx_node+1, len(ind)) #[1:]

                    # avoid repeated hashes (this is good for statistics)
                    if len([i for i in self.pop_hash[closest_hash]
                            if str(i) == str(ind_subtree)]) == 0:
                        
                        self.n_new_hashes += 1
                        bisect.insort(
                            self.pop_hash[closest_hash], ind_subtree,
                            key=lambda ind: self.key(ind))
                        
                else: # Making it a new semantics item
                    if binary_hash not in self.pop_hash:
                        self.pop_hash[binary_hash] = [ind_subtree]
                        self.n_new_hashes += 1
                        self.lsh.index( h )

            assert original_str == str(pop[idx])
            
            new_pop.append(ind)
            if refit_ind:
                refit.append(idx)

        if replace_pop:
            return new_pop, refit
        
        self.n_simplifications=0# ignoring simplifications, as we are returning original pop
        return pop, []