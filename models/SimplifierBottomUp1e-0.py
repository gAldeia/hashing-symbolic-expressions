from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'              : True,
          'simplification_method' : 'bottom_up',
          'simplification_tolerance' : 1e-0   }
    }
) 

name = "Bottom Up 1e-0"
