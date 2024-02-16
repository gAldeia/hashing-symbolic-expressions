from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'smart_variation'       : True,
          'simplify'              : False,
          'simplification_method' : 'bottom_up',
          'use_mab'               : True }
    }
) 

name = "LSH Variator with MAB"
