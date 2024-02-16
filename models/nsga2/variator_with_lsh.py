# Hash based variation operator
import lshashpy3 as lshash
import bisect
import warnings
import numpy as np

from functools import partial
import operator
from .deap_utils import get_complexity
from deap import base, creator, tools, gp

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

# implements variation operators with LSH
class HashVariator():
    def __init__(self, Individual, toolbox, rnd_generator, hash_len=256):
        self.toolbox       = toolbox
        self.Individual    = Individual
        self.rnd_generator = rnd_generator
        self.hash_len      = hash_len
        self.distance_func = "l1norm"
        self.num_hashtables = 5


    def _is_equal(self, ind1, ind2):
        if len(ind1) != len(ind2):
            return False
        
        for node1, node2 in zip(ind1[:], ind2[:]):
            # If it is a constant, we dont care about the value
            if  isinstance(type(node1), gp.MetaEphemeral) \
            and isinstance(type(node2), gp.MetaEphemeral) :
                continue
            elif node1.name == node2.name:
                continue
            else:
                return False

        return True
    

    def _predict_hash(self, ind, X, y):
        # auxiliary function to compile an individual and evaluate it on the data

        expr = self.toolbox.compile(expr=ind)
        pred = np.array([expr(*x) for x in X])

        # Approach 1
        #normalized_pred = zscore(pred) + 1e-8

        #return normalized_pred
        # Approach 2
        #scaled_pred = MinMaxScaler(feature_range=(-1, 1)).fit_transform(pred.reshape(-1, 1))[:, 0]

        # Approach 3
        # first we de-trend the data. then we add 1e-8, so positive and negative
        # constants are mapped to the same hash
        stardized_pred = (pred - np.mean(pred)) + 1e-8

        # This makes the hash space have more colisions based on direction of predictions
        thresholded = np.where(stardized_pred > 0, 1, -1)

        return thresholded 


    def init(self, pset, X, y):
        print(f"hashtable will have dimensions {self.hash_len} x {len(y)}")

        self.pset = pset

        # colision is what determines how we explore the search space.
        # more hashtables can help mitigate the stochastic mapping
        self.lsh = lshash.LSHash(self.hash_len, len(y), num_hashtables=self.num_hashtables)
        self.pop_hash = {}

        dummy_ind = self.Individual([pset.mapping['rand100']()])
        dummy_ind[0].value = 1.0 # setting the value to be zero, same as constant nodes would have

        h = self._predict_hash(dummy_ind, X, y)

        # Starting to store the individual
        pop_index = str(len(self.pop_hash))
        self.pop_hash[pop_index] = [dummy_ind]
        print(f"starting to index. {len(h)}, {pop_index}")

        assert pop_index is not None, "trying to index with None"
        self.lsh.index( input_point=h, extra_data=pop_index )

        for i in range(X.shape[1]):
            feature_ind = self.Individual([pset.mapping[f'ARG{i}']])

            h = self._predict_hash(feature_ind, X, y)

            pop_index = str(len(self.pop_hash))
            self.pop_hash[pop_index] = [feature_ind]
            print(f"starting to index. {len(h)}, {pop_index}")
            
            assert pop_index is not None, "trying to index with None"
            self.lsh.index( input_point=h, extra_data=pop_index )
                
        for index, individuals in self.pop_hash.items():
            index = str(index)

            h = self._predict_hash(individuals[0], X, y)
            val, d = self.lsh.query(h)[0]
            h, pop_index = val

            # print(f"{index} (pop_index={pop_index}, d={d}): {str(individuals[0])}")

            assert len(individuals) == 1, "two individuals in the same list"
            assert pop_index == index, "stored with mismatching index"
            assert isinstance(pop_index, str), "key to pop_hash is nos str"
            # assert d == 0, "failed to retrieve exact hash"

        print(f"initialized {len(self.pop_hash.keys())} keys")

        self.X_ = X
        self.y_ = y

        return self
    

    def _memoize_and_find_spots(self, ind, constants_not_allowed=False):
        # it doesnt matter the order we simplify.
        # only subtrees that does not eval to nan will be stored and used to replace
        candidate_idxs = []

        # First we iterate over the tree to get more information to sample the mutation
        for idx_node, node in enumerate(ind):

            # mutation: we dont want to replace a const with anything else of similar hash (it would be irrelevant or bloated const)
            if isinstance(type(node), gp.MetaEphemeral) \
            and constants_not_allowed:
                continue

            ind_subtree = ind[ind.searchSubtree(idx_node)]

            ind_subtree = self.Individual(self.toolbox.clone(ind_subtree)[:])
            h = self._predict_hash(ind_subtree, self.X_, self.y_)

            if not np.all(np.isfinite(h)):
                continue

            # We may have the hash memoized in all hashtables, so we update accordingly
            results = self.lsh.query(h, num_results=self.num_hashtables, distance_func=self.distance_func)
            
            # We will memoize a new subtree
            if len(results)==0:
                # ((input vector, pop_index), distance)
                results = [( (h, str(len(self.pop_hash))), 0.0)]

            # Results contains hashes ordered by distance. We'll try to put copies 
            # into every hash that has distance zero
            append = False
            for res in results:
                # print(res)
                data, d      = res
                h, pop_index = data

                # We dont memoize constants (but we let them to be changed in cx)
                if pop_index == "0":
                    continue

                assert pop_index is not None, "Binary hash wasnt calculated."

                if d == 0.0:
                    if pop_index not in self.pop_hash:
                        self.lsh.index( input_point=h, extra_data=pop_index )
                        self.pop_hash[pop_index] = []

                    if not any([self._is_equal(ind_subtree, pop_subtree) 
                                for pop_subtree in self.pop_hash[pop_index]]):
                        self.pop_hash[pop_index].append(ind_subtree)
                    else:
                        # print(str(ind_subtree), "already in candidates:")
                        # for others in self.pop_hash[pop_index]:
                        #     print("\t -", str(others))
                        pass

                    if len(self.pop_hash[pop_index])>1:
                        # print(f"starting to index. {len(h)}, {pop_index}")
                        append=True
                else:
                    # We can stop, because results is ordered by distance
                    break
    
            if append:
                candidate_idxs.append(idx_node) 

        return candidate_idxs
    

    def mutate(self, ind):
        # print("mut")
        # print("original:", str(ind))

        # ind is alredy a clone
        candidate_idxs = self._memoize_and_find_spots(ind, constants_not_allowed=True)

        if len(candidate_idxs) == 0:
            # print("no candidates in mut")
            return (ind, )
        else:
            # print(f"len keys: {len(self.pop_hash)}. total {sum(len(v) for k, v in self.pop_hash.items())}")
            
            # Now we pick a random spot, get its hash, and change the subtree with any from the same hash
            spot_idx     = candidate_idxs[self.rnd_generator.choice(len(candidate_idxs))]
            spot_subtree = self.Individual(ind[ind.searchSubtree(spot_idx)])
            spot_vector  = self._predict_hash(spot_subtree, self.X_, self.y_)

            # print(f"index was {spot_idx}")
            # print(f"spot vector: {spot_vector[:3]}")

            # Closest hash
            # Should always work, because memoize_and_find_spots worked with the subtree semantics
            res = self.lsh.query(spot_vector, num_results=1, distance_func=self.distance_func)
            spot_vector, pop_index = res[0][0]

            # Same hash
            replacements = [i for i in range(len(self.pop_hash[pop_index]))
                            if not self._is_equal(self.pop_hash[pop_index][i], spot_subtree)]
            
            spot_replace = self.pop_hash[pop_index][
                self.rnd_generator.choice(replacements)]
            
            new_ind = self.Individual( ind[:spot_idx] + \
                self.toolbox.clone(spot_replace)[:]   + \
                ind[spot_idx+len(spot_subtree):]        )

            # print("new:     ", str(new_ind))

            # print("\treplacements:" )
            # for others in self.pop_hash[pop_index][:5]:
            #     print("\t -", str(others))

            return (new_ind, )


    def subtree(self, ind):
        xmen, = gp.mutUniform(ind, pset=self.pset, expr=self.toolbox.expr)
        
        self._memoize_and_find_spots(xmen, constants_not_allowed=True)

        return xmen,
        

    def cross(self, ind1, ind2):
        # print("cross")
        offs = []
        for ind in [ind1, ind2]:
            # print("original:", str(ind))

            # ind is alredy a clone
            _ = self._memoize_and_find_spots(ind)

            # Every node is a candidate (we wont be using its hash to get a new subtree)
            candidate_idxs = list(range(len(ind)))

            # print(f"len keys: {len(self.pop_hash)}. total {sum(len(v) for k, v in self.pop_hash.items())}")

            # Now we pick a random spot, get its hash, and change the subtree with any from the same hash
            spot_idx     = candidate_idxs[self.rnd_generator.choice(len(candidate_idxs))]
            spot_subtree = self.Individual(ind[ind.searchSubtree(spot_idx)])
            spot_vector  = self._predict_hash(spot_subtree, self.X_, self.y_)

            # print(f"index was {spot_idx}")
            # print(f"spot vector: {spot_vector[:3]}")

            # Closest hash
            res = self.lsh.query(
                spot_vector, num_results=1, distance_func=self.distance_func)
            spot_vector, not_pop_index = res[0][0]

            # Any other hash
            pop_index = not_pop_index
            while pop_index == not_pop_index:
                pop_index = str(self.rnd_generator.choice(len(self.pop_hash)))

            spot_replace = self.pop_hash[pop_index][
                self.rnd_generator.choice(len(self.pop_hash[pop_index]))]
            
            new_ind = self.Individual( ind[:spot_idx] + \
                self.toolbox.clone(spot_replace)[:]   + \
                ind[spot_idx+len(spot_subtree):]        )

            # print("new:     ", str(new_ind))
            
            # print("\treplacements:" )
            # for h in self.pop_hash[pop_index][:5]:
            #     print("\t -", str(h))

            offs.append(new_ind)

        return offs