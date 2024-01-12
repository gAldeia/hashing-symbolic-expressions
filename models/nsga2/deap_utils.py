# Auxiliary files to do some GA stuff with DEAP.
# Created by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

import numpy as np
from deap import gp


def PTC2_deap(pset, min_, max_, max_size_sampler, type_=None):
    # Function to emulate PTC2 inside DEAP
    
    # From deap docs: 
    # The tree is built from the root to the leaves, and it stops growing
    # the current branch when the *condition* is fulfilled: in which case, it
    # back-tracks, then tries to grow another branch until the *condition* is
    # fulfilled again, and so on. 

    # Inserting this custom fields in pset so we can access inside
    # condition when generating expressions
    pset.local_size_counter = 0
    max_size = max_size_sampler()

    # The condition is a function that takes two arguments,
    # the height of the tree to build and the current
    # depth in the tree.
    def condition(height, depth):
        pset.local_size_counter += 1
        return depth>height or pset.local_size_counter>max_size

    return gp.generate(pset, min_, max_, condition, type_)


# Ephemeral random constants (this is how we add constants into our trees)
_deap_random = np.random.default_rng()

def ERC1(): return _deap_random.uniform(-1,1)
def ERC100(): return _deap_random.uniform(-100,100)

# Let's protect the math
def cdiv(left, right): return (left / (np.sqrt(1 + (right*right))))
def sqrtabs(x): return np.sqrt(np.abs(x))

# Extending function options to  have n-ary versions of commutative functions.
# These also work as examples on how to implement reducers e.g. min, max, median
def add3(left, middle, right): return left + middle + right
def mul3(left, middle, right): return left * middle * right

def add4(a, b, c, d): return a+b+c+d
def mul4(a, b, c, d): return a*b*c*d

# OBS: if you create new functions, remember to define a complexity to it

# This is how we import what we implemented here
node_functions = { # ( name (str), (function, arity) )
    # Arithmetic
    'div' : (cdiv, 2), #division without discontinuity
    'add' : (np.add, 2),
    'sub' : (np.subtract, 2),
    'mul' : (np.multiply, 2),
    
    # n-ary commutative nodes
    'add3' : (add3, 3),
    'add4' : (add4, 4),
    'mul3' : (mul3, 3),
    'mul4' : (mul4, 4),

    # Simple reducers
    'maximum' : (np.maximum, 2),
    'minimum' : (np.minimum, 2),

    # Trigonometric
    'sin' : (np.sin, 1),
    'cos' : (np.cos, 1),
    'tan' : (np.tan, 1),

    'sqrtabs' : (sqrtabs, 1),
    'log1p'   : (np.log1p, 1),
    'exp'     : (np.exp, 1),
    'square'  : (np.square, 1),
    'abs'     : (np.abs, 1),
}

# Complexity calculation based on pre-defined values.
# How to use the dict for a node: `operator_complexities.get(node.name, 1)`
# (there's no key for features, as they may have any number n in ARGn. The 
# complexity for a feature is 1, so it should be the default value if the key
# is not in the dict. Yes  I know, it could be better, but I dont want to 
# use regex or auxiliary functions to access this dict and handle cases like
# ARG1, ARG9, ARG0, ARG128, ARG19823616745, etc).
operator_complexities = {
    'cos': 5,
    'sin': 5,
    'tan': 5,

    'sqrtabs': 4,
    'exp'    : 4,
    'log1p'  : 8,
    'square' : 3,
    'abs'    : 3,

    'minimum': 3,
    'maximum': 3,
    
    'add': 2,
    'sub': 2,
    'mul': 3,
    'div': 4,

    # Linear progression based on arity
    'add3': 4, 
    'add4': 6,
    'mul3': 6,
    'mul4': 9,

    'rand100': 2, # our ERC
}

def nth_child(ind, idx, n):
    # We assume ind is already the root
    if ind[idx].arity==0 or n>ind[idx].arity:
        return None
    
    # index where the first child starts
    nth_idx = idx+1 

    # Incrementing for subsequent children
    for _ in range(n):
        child_range = ind.searchSubtree(nth_idx)
        nth_idx += len( ind[child_range] )

    return nth_idx

def get_complexity(ind, curr_idx=None):
    if curr_idx is None:
        curr_idx = 0

    root = ind[curr_idx]
    node_complexity = operator_complexities.get(root.name, 1)

    children_complexity_sum = 0 # accumulator for children complexities
    for child_number in range(root.arity):
        child_idx = nth_child(ind, curr_idx, child_number)
        children_complexity_sum += get_complexity(ind, child_idx)

    # avoid multiplication by zero if the node is a terminal
    children_complexity_sum = max(children_complexity_sum, 1)

    return node_complexity*children_complexity_sum
