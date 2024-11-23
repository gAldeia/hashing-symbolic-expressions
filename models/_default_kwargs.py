kwargs = {
    'pop_size'       : 100, 
    'max_gen'        : 101,
    'max_depth'      : 10,
    'max_size'       : 128,
    'objectives'     : ['error', 'size'],
    'cx_prob'        : 1/(4+1), #1/(4+1), # n_mutations+1
    'initialization' : 'uniform',
    'pick_criteria'  : 'error',
    'validation_size': 0.33,
    'verbosity'      : 2,
    'survival'       : 'tournament', #'nsga2', 'offspring', 'tournament'
    'simplification_tolerance' : 1e-2,
    'functions'      : [
                        'div', 'add', 'sub', 'mul',
                        # 'add3', 'add4', 'mul3', 'mul4',
                        'maximum', 'minimum',
                        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sqrt', 
                        'sqrtabs', 'log1p', 'expm1', 'log', 'exp', 'square', 'abs'
                       ],
}