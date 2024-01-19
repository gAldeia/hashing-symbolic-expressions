kwargs = {
    'pop_size'       : 48, 
    'max_gen'        : 150, #255,
    'max_depth'      : 7,
    'max_size'       : 64,
    'objectives'     : ['error', 'size'],
    'cx_prob'        : 1/(4+1), # n_mutations+1
    'initialization' : 'uniform',
    'pick_criteria'  : 'error',
    'validation_size': 0.33,
    'verbosity'      : 1,
    'functions'      : [
                            'div', 'add', 'sub', 'mul',
                            #'add3', 'add4', 'mul3', 'mul4',
                            'maximum', 'minimum',
                            'sin', 'cos', 'tan',
                            'sqrtabs', 'log1p', 'log', 'exp', 'square', 'abs'
                       ],
}