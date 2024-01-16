kwargs = {
    'pop_size'              : 100, 
    'max_gen'               : 100,
    'max_depth'             : 6,
    'max_size'              : 64,
    'objectives'            : ['error', 'size'],
    'cx_prob'               : 1/(4+1), # n_mutations+1
    'initialization'        : 'uniform',
    'pick_criteria'         : 'error',
    'validation_size'       : 0.33,
    'verbosity'             : 1,
}