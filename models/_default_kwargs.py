kwargs = {
    'pop_size'       : 150, 
    'max_gen'        : 255,
    'max_depth'      : 10,
    'max_size'       : 2**6,
    'objectives'     : ['error', 'size'],
    'cx_prob'        : 0.5, #1/(4+1), # n_mutations+1
    'initialization' : 'uniform',
    'pick_criteria'  : 'error',
    'validation_size': 0.3,
    'verbosity'      : 0,
    'survival'       : 'nsga2',
    'simplification_tolerance' : 1e-10,
    'functions'      : [
                        'div', 'add', 'sub', 'mul',
                        'add3', 'add4', 'mul3', 'mul4',
                        'maximum', 'minimum',
                        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 
                        'sqrtabs', 'log1p', 'expm1', 'log', 'exp', 'square', 'abs'
                       ],
}