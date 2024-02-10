from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'smart_variation'       : True,
          'simplify'              : True,
          'simplification_method' : 'bottom_up' }
    }
) 

name = "LSH Variator with simplify"
