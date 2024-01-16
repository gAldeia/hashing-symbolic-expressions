from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'              : True,
          'simplification_method' : 'bottom_up',
           'simplify_only_last'   : True      }
    }
) 

name = "Bottom Up (only last)"
