kwargs = {
    'pop_size'              : 100, 
    'max_gen'               : 500,
    'max_depth'             : 10,
    'max_size'              : 100,
    'objectives'            : ['error', 'size'],
    'cx_prob'               : 1/7,
    'initialization'        : 'uniform',
    'pick_criteria'         : 'error',
    'validation_size'       : 0.33,
    'simplify'              : True,
    'simplification_method' : "bottom_up",
    'verbosity'             : False,
}