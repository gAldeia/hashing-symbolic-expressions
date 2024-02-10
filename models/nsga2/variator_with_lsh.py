# Hash based variation operator
import lshashpy3 as lshash
import bisect
import warnings
import numpy as np

from functools import partial
import operator
from .deap_utils import get_complexity
from deap import base, creator, tools, gp


# implements variation operators with LSH
class HashVariator():
    def __init__(self, Individual, toolbox, rnd_generator, hash_len=128):
        self.toolbox       = toolbox
        self.Individual    = Individual
        self.rnd_generator = rnd_generator
        self.hash_len      = hash_len


    def _predict_hash(self, ind, X, y):
        # auxiliary function to compile an individual and evaluate it on the data

        expr = self.toolbox.compile(expr=ind)
        pred = np.array([expr(*x) for x in X])

        # constant predictions, should be mapped into dummy_ind hash
        if np.std(pred) <= 1e-10:
           return np.ones_like(pred)
        
        return pred
    

    def init(self, pset, X, y):
        # colision is what determines how we explore the search space
        self.lsh= lshash.LSHash(self.hash_len, len(y))
        self.pop_hash = {}

        for i in range(X.shape[1]):
            feature_ind = self.Individual([pset.mapping[f'ARG{i}']])

            h = self._predict_hash(feature_ind, X, y)
            self.lsh.index(h)

            binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
            self.pop_hash[binary_hash] = [feature_ind]

        dummy_ind = self.Individual([pset.mapping['rand100']()])
        dummy_ind[0].value = 0.0 # setting the value to be zero, same as constant nodes would have

        h = self._predict_hash(dummy_ind, X, y)
        self.lsh.index(h)

        binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)

        self.pop_hash[binary_hash] = [dummy_ind]

        self.X_ = X
        self.y_ = y

        return self

    def _memoize_and_find_spots(self, ind):
        # it doesnt matter the order we simplify.
        # only subtrees that does not eval to nan will be stored and used to replace
        candidate_idxs = []

        # First we iterate over the tree to get more information to sample the mutation
        for idx_node, node in list(enumerate(ind))[::-1]:
            ind_subtree = ind[ind.searchSubtree(idx_node)]

            if len(ind_subtree) <=1:
                candidate_idxs.append(idx_node)
                continue

            ind_subtree = self.Individual(self.toolbox.clone(ind_subtree)[:])
            h = self._predict_hash(ind_subtree, self.X_, self.y_)

            if np.any(np.bitwise_not(np.isfinite(h))):
                continue

            binary_hash = self.lsh._hash(self.lsh.uniform_planes[0], h)
            candidate_idxs.append(idx_node) 

            if binary_hash not in self.pop_hash:
                self.pop_hash[binary_hash] = [ind_subtree]
                self.lsh.index( h )
            else:
                self.pop_hash[binary_hash].append(ind_subtree)

        return candidate_idxs
    

    def mutate(self, ind):
        # ind is alredy a clone
        candidate_idxs = self._memoize_and_find_spots(ind)

        # Now we pick a random spot, get its hash, and change the subtree with any from the same hash
        spot_idx     = candidate_idxs[self.rnd_generator.choice(len(candidate_idxs))]
        spot_subtree = self.Individual(ind[ind.searchSubtree(spot_idx)])
        spot_vector  = self._predict_hash(spot_subtree, self.X_, self.y_)
        spot_hash    = self.lsh._hash(self.lsh.uniform_planes[0], spot_vector)
        spot_replace = self.pop_hash[spot_hash][self.rnd_generator.choice(len(self.pop_hash[spot_hash]))]
        
        new_ind = self.Individual( ind[:spot_idx] + \
            self.toolbox.clone(spot_replace)[:]   + \
            ind[spot_idx+len(spot_subtree):]        )

        # print("original:", str(ind), "new:", str(new_ind))

        return (new_ind, )

    def cross(self, ind1, ind2):
        offs = []
        for ind in [ind1, ind2]:
            # ind is alredy a clone
            candidate_idxs = self._memoize_and_find_spots(ind)

            # Now we pick a random spot, get its hash, and change the subtree with any from the same hash
            spot_idx     = candidate_idxs[self.rnd_generator.choice(len(candidate_idxs))]
            spot_subtree = self.Individual(ind[ind.searchSubtree(spot_idx)])
            
            # Any other hash
            spot_hash = self.rnd_generator.choice(list(self.pop_hash.keys()))

            spot_replace = self.pop_hash[spot_hash][self.rnd_generator.choice(len(self.pop_hash[spot_hash]))]
            
            new_ind = self.Individual( ind[:spot_idx] + \
                self.toolbox.clone(spot_replace)[:]   + \
                ind[spot_idx+len(spot_subtree):]        )

            # print("original:", str(ind), "new:", str(new_ind))

            offs.append(new_ind)

        return offs