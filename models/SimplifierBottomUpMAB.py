from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'              : True,
          'simplification_method' : 'bottom_up',
           'use_mab'              : True }
    }
) 

name = "Bottom Up MAB"
