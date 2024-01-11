# Implement a hash-based "simplify" function to be imported by nsga2_deap
# Created by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

# https://github.com/loretoparisi/lshash
import lshashpy3 as lshash
import bisect

import numpy as np

class HashSimplifier:
    def __init__(self, Individual, Fitness, toolbox, hash_len=32, tolerance=1e-6):
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

        return pred - np.mean(pred)
    

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
        
        # Counters for statistics
        self.n_simplifications = 0
        self.n_new_hashes      = 0

        self.initialized = True


    def simplify_pop_bottom_up(self, pop, X, y):

        self.n_simplifications = 0
        self.n_new_hashes      = 0

        new_pop = []
        for idx, ind in enumerate(pop):
            indexes = list(enumerate(ind))[-1:0:-1]
            for idx_node, node in indexes:
                ind_subtree = ind[ind.searchSubtree(idx_node)]

                if len(ind_subtree) <=1:
                    continue

                ind_subtree = self.Individual(ind_subtree)

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)

                # Hash for the individual
                binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
                
                # Querying for the closest result
                res = self.lsh.query(h, num_results=1, distance_func="euclidean")
                
                try:
                    ((v, extra), d) = res[0]
                    binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], v)
                except IndexError:
                    self.n_new_hashes += 1
                    self.lsh.index( h )
                    self.pop_hash[binary_hash] = [ind_subtree]

                    continue

                if d <= self.tolerance * len(y): # they are similar
                    bisect.insort(
                        self.pop_hash[binary_hash], ind_subtree,
                        key=lambda i: len(i))
                    
                    index_smlst = 0

                    if len(self.pop_hash[binary_hash][index_smlst]) \
                    <  len(ind):
                        # Several simplifications can happen in a single individual
                        self.n_simplifications += 1

                        ind = ind[:idx_node] + \
                                self.pop_hash[binary_hash][index_smlst] + \
                                ind[idx_node+len(ind_subtree):]

                        # We need to wrap here to allow the loop to continue
                        # (as it uses some of PrimitiveTree methods)
                        ind = self.Individual(ind)
                    
                else: # Making it a new semantics item
                    self.n_new_hashes += 1
                    self.lsh.index( h )
                    self.pop_hash[binary_hash] = [ind_subtree]

            # Wrapping the list into the individual again
            new_ind = self.Individual(ind)
            new_ind.fitness = self.Fitness()
            new_ind.fitness.values = pop[idx].fitness.values
            new_pop.append(new_ind)

        return new_pop
    
    
    def simplify_pop_top_down(self, pop, X, y):

        new_pop = []
        for idx, ind in enumerate(pop):

            indexes = range(1, len(ind))[1:]
            while len(indexes)>1:
                idx_node = indexes[0]
                indexes = indexes[1:]

                ind_subtree = ind[ind.searchSubtree(idx_node)]

                if len(ind_subtree) <=1:
                    continue

                ind_subtree = self.Individual(ind_subtree)

                # Semantics to be hashed
                h = self._predict_hash(ind_subtree, X, y)

                # Hash for the individual
                binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
                
                # Querying for the closest result
                res = self.lsh.query(h, num_results=1, distance_func="euclidean")
                
                try:
                    ((v, extra), d) = res[0]
                    binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], v)
                except IndexError:
                    self.n_new_hashes += 1
                    self.lsh.index( h )
                    self.pop_hash[binary_hash] = [ind_subtree]

                    continue

                if d <= self.tolerance * len(y): # they are similar
                    bisect.insort(
                        self.pop_hash[binary_hash], ind_subtree,
                        key=lambda i: len(i))
                    
                    index_smlst = 0

                    if len(self.pop_hash[binary_hash][index_smlst]) \
                    <  len(ind):
                        # Several simplifications can happen in a single individual
                        self.n_simplifications += 1
                        
                        ind = ind[:idx_node] + \
                                self.pop_hash[binary_hash][index_smlst] + \
                                ind[idx_node+len(ind_subtree):]

                        # We need to wrap here to allow the loop to continue
                        # (as it uses some of PrimitiveTree methods)
                        ind = self.Individual(ind)

                        # If we simplified, then we need to update the index list
                        indexes = range(idx_node, len(ind))[1:]

                else: # Making it a new semantics item
                    self.n_new_hashes += 1
                    self.lsh.index( h )
                    self.pop_hash[binary_hash] = [ind_subtree]

            # Wrapping the list into the individual again
            new_ind = self.Individual(ind)
            new_ind.fitness = self.Fitness()
            new_ind.fitness.values = pop[idx].fitness.values
            new_pop.append(new_ind)

        return new_pop