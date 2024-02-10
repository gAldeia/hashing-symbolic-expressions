from  .nsga2.estimator import NSGAIIRegressor
from ._default_kwargs import kwargs

reg = NSGAIIRegressor(
    **{**kwargs,
       **{'simplify'        : False,
          'smart_variation' : True}
    }
) 

name = "LSH Variator"
