# Implement a hash-based "simplify" function to be imported by nsga2_deap
# Created by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

# https://github.com/loretoparisi/lshash
import lshashpy3 as lshash
import bisect

import numpy as np

class HashSimplifier:
    def __init__(self, Individual, Fitness, toolbox,
                 hash_len=32, tolerance=1e-15):
        self.Individual = Individual
        self.Fitness = Fitness
        self.toolbox = toolbox
        self.hash_len=hash_len # number of bits used for the hash
        self.tolerance=tolerance # the distance tolerance to consider two individual the same
        self.initialized = False
    

    def _predict_hash(self, ind, X, y):
        # auxiliary function to compile an individual and evaluate it on the data

        expr = self.toolbox.compile(expr=ind)
        pred = np.array([expr(*x) for x in X])

        return pred # - np.mean(pred)
    

    def initialize(self, pset, X, y):
        # Use the first individual to extract info to initialize the table

        dummy_ind = self.Individual([pset.mapping['rand100']()])
        dummy_ind[0].value = np.mean(y) # setting the value to be mean prediction

        h = self._predict_hash(dummy_ind, X, y)

        # Creating our locality sensitive hashing table
        self.lsh = lshash.LSHash(self.hash_len, len(y))
        
        # Setting the first index
        self.lsh.index(h)

        # getting the hash representation for the prediction vector 'h'
        binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)

        # This is the memoization part
        # Adding the first value into the memoization
        self.pop_hash = {binary_hash : [dummy_ind]}
        
        # Now do it again for each terminal
        for i in range(X.shape[1]):
            feature_ind = self.Individual([pset.mapping[f'ARG{i}']])
            h = self._predict_hash(feature_ind, X, y)
            self.lsh.index(h)
            binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
            self.pop_hash[binary_hash] = [feature_ind]

        # Counters for statistics
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        assert 0 < len(self.pop_hash.keys()) <= X.shape[1]+1

        self.initialized = True


    def simplify_pop_bottom_up(self, pop, X, y, replace_pop=True):
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        new_pop = []
        for idx, ind in enumerate(pop):
            # print(f'simplifying ind idx{idx} {ind}')
            original_str = str(ind)
            
            indexes = list(enumerate(ind))[::-1]
            # print(f' - list of indexes {indexes}')
            for idx_node, node in indexes:
                # print(f'   - {idx_node}, {node.name}')
                ind_subtree = ind[ind.searchSubtree(idx_node)]
                # print(f'   - subtree {ind_subtree}')

                if len(ind_subtree) <=1:
                    # print(f'     - skipping')
                    continue

                ind_subtree = self.Individual(ind_subtree)
                # print(f'     - cast into ind {ind_subtree}')

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)
                # print(f'     - semantics {h[:3]}')

                if np.all(np.isnan(h)):
                    # print(f'     - bad semantics')
                    continue

                # Hash for the individual
                binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
                # print(f'     - hash {binary_hash}')

                # Querying for the closest result
                res = self.lsh.query(h, num_results=1, distance_func="euclidean")
                
                closest_hash = None
                try: 
                   # # print(f'       - query {res}')
                    ((v, extra), d) = res[0]
                    closest_hash = self.lsh._hash(self.lsh.uniform_planes[0], v)
                    # print(f'       - closest hash {closest_hash} with dist {d} and semantics {v[:3]}')
                    # print(f'       - ind is {self.pop_hash[closest_hash][0]}')
                except IndexError:
                    # print(f'       - indexing it (failed to find any hash)')

                    # Stop inserting thrash into the dict
                    # self.n_new_hashes += 1
                    # self.lsh.index( h )
                    # if binary_hash not in self.pop_hash:
                    #     self.pop_hash[binary_hash] = [ind_subtree]
                    # else:
                    #     bisect.insort(
                    #         self.pop_hash[binary_hash], ind_subtree,
                    #         key=lambda ind: len(ind))

                    continue

                # print(f'       - tolerance check {d} <= {self.tolerance }')    
                if d <= self.tolerance : # they are similar
                    # print(f'         - success, trying to simplify')
                
                    bisect.insort(
                        self.pop_hash[closest_hash], ind_subtree,
                        key=lambda ind: len(ind))
                    
                    if len(self.pop_hash[closest_hash][0]) < len(ind_subtree):
                        
                        # print(f'         - can be simplified cause len {len(self.pop_hash[closest_hash][0])} of {self.pop_hash[closest_hash][0]} < len {len(ind_subtree)} of {ind_subtree}')
                        
                        # Several simplifications can happen in a single individual
                        self.n_simplifications += 1

                        ind = ind[:idx_node] + \
                                self.pop_hash[closest_hash][0] + \
                                ind[idx_node+len(ind_subtree):]

                        # We need to wrap here to allow the loop to continue
                        # (as it uses some of PrimitiveTree methods)
                        ind = self.Individual(ind)
                        # print(f'         - result {ind}')
                    else:
                        # print(f'         - CANNOT be simplified cause len {len(self.pop_hash[closest_hash][0])} of {self.pop_hash[closest_hash][0]} >= len {len(ind_subtree)} of {ind_subtree}')
                        pass

                else: # Making it a new semantics item
                    # print(f'         - not simplifiable, will insert it')
                    self.n_new_hashes += 1
                    self.lsh.index( h )
                    if not (binary_hash in self.pop_hash):
                        self.pop_hash[binary_hash] = [ind_subtree]
                    else:
                        bisect.insort(
                            self.pop_hash[binary_hash], ind_subtree,
                            key=lambda ind: len(ind))

            assert original_str == str(pop[idx])

            # Wrapping the list into the individual again
            new_ind = self.Individual(ind)
            new_ind.fitness = self.Fitness()
            new_ind.fitness.values = pop[idx].fitness.values
            # print(f'  - final result {new_ind}')

            new_pop.append(new_ind)

        if replace_pop:
            return new_pop
        
        self.n_simplifications=0# ignoring simplifications, as we are returning original pop
        return pop
    

    def simplify_pop_top_down(self, pop, X, y, replace_pop=True):
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        new_pop = []
        for idx, ind in enumerate(pop):

            original_str = str(ind)

            indexes = range(0, len(ind))
            while len(indexes)>1:
                # print( indexes[0], len(indexes), len(ind))
                idx_node = indexes[0]
                indexes  = indexes[1:]

                ind_subtree = ind[ind.searchSubtree(idx_node)]

                if len(ind_subtree) <=1:
                    continue

                ind_subtree = self.Individual(ind_subtree)

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)

                if np.all(np.isnan(h)):
                    continue

                # Hash for the individual
                binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
                
                # Querying for the closest result
                res = self.lsh.query(h, num_results=1, distance_func="euclidean")
                
                closest_hash = None
                try:
                    ((v, extra), d) = res[0]
                    closest_hash = self.lsh._hash(self.lsh.uniform_planes[0], v)
                except IndexError:
                    # self.n_new_hashes += 1
                    # self.lsh.index( h )
                    # if binary_hash not in self.pop_hash:
                    #     self.pop_hash[binary_hash] = [ind_subtree]
                    # else:
                    #     bisect.insort(
                    #         self.pop_hash[binary_hash], ind_subtree,
                    #         key=lambda ind: len(ind))

                    continue

                if d <= self.tolerance : # they are similar
                    bisect.insort(
                        self.pop_hash[closest_hash], ind_subtree,
                        key=lambda ind: len(ind))
                    
                    if len(self.pop_hash[binary_hash][0]) < len(ind_subtree):
                        # Several simplifications can happen in a single individual
                        self.n_simplifications += 1
                        
                        ind = ind[:idx_node] + \
                                self.pop_hash[binary_hash][0] + \
                                ind[idx_node+len(ind_subtree):]

                        # We need to wrap here to allow the loop to continue
                        # (as it uses some of PrimitiveTree methods)
                        ind = self.Individual(ind)

                        # If we simplified, then we need to update the index list
                        # (we also skip the first)
                        indexes = range(idx_node+1, len(ind)) #[1:]

                else: # Making it a new semantics item
                    self.n_new_hashes += 1
                    self.lsh.index( h )
                    if binary_hash not in self.pop_hash:
                        self.pop_hash[binary_hash] = [ind_subtree]
                    else:
                        bisect.insort(
                            self.pop_hash[binary_hash], ind_subtree,
                            key=lambda ind: len(ind))
            
            assert original_str == str(pop[idx])
            
            # Wrapping the list into the individual again
            new_ind = self.Individual(ind)
            new_ind.fitness = self.Fitness()
            new_ind.fitness.values = pop[idx].fitness.values
            new_pop.append(new_ind)

        if replace_pop:
            return new_pop
        
        self.n_simplifications=0# ignoring simplifications, as we are returning original pop
        return pop