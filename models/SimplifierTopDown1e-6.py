from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'              : True,
          'simplification_method' : 'top_down',
          'simplification_tolerance' : 1e-6   }
    }
) 

name = "Simplifier Top Down 1e-6"
