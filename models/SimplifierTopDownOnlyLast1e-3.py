from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'              : True,
          'simplification_method' : 'top_down',
           'simplify_only_last'   : True,
           'simplification_tolerance' : 1e-3 }
    }
) 

name = "Top Down 1e-3 (only last)"
