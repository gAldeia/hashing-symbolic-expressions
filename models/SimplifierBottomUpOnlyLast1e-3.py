from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'                 : True,
          'simplification_method'    : 'bottom_up',
          'simplification_tolerance' : 1e-3,
          'simplify_only_last'       : True      }
    }
) 

name = "Bottom Up 1e-3 (only last)"
