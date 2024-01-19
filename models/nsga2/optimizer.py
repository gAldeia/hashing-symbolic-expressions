# Yes, we can levenberg-marquardt DEAP trees, but things get nasty pretty fast..
# Modified by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

import numpy as np
from scipy.optimize import least_squares

from deap import gp

def optimize_individual(toolbox, ind, X, y):

    # Getting original coeffs
    coefs = []
    for i, node in enumerate(ind):
        if isinstance(node, gp.Terminal):
            if isinstance(type(node), gp.MetaEphemeral):
                coefs.append(ind[i].value)

                # Creating new instance to avoid collateral effects
                ind[i] = type(node)()
    
    if len(coefs) == 0: # nothing to do here
        return ind
    
    # Function to update the coeffs following order they occur, then return prediction
    def model(params, xs):
        if np.sum(np.isfinite(params)) < len(params):
            raise ValueError("Coefficients are not finite")

        c_idx = 0
        for i, node in enumerate(ind):
            if isinstance(node, gp.Terminal):
                if isinstance(type(node), gp.MetaEphemeral):
                    ind[i].value = params[c_idx]
                    c_idx += 1

        expr = toolbox.compile(expr=ind)
        pred = np.array([expr(*x) for x in xs])

        return pred

    def residuals(params, x, y):
        return y - model(params, x)

    params_initial   = coefs
    optimized_params = params_initial

    try:
        result = least_squares(residuals, params_initial, args=(X, y), max_nfev=10)
        optimized_params = result.x
    except ValueError: # Optimization failed --- theres inf in the data
        pass
    except Exception as e:
        print(ind)
        raise(e)

    # Updating coeffs to final values
    c_idx = 0
    for i, node in enumerate(ind):
        if isinstance(node, gp.Terminal):
            if isinstance(type(node), gp.MetaEphemeral):
                ind[i].value = optimized_params[c_idx]
                c_idx += 1
                
    return ind