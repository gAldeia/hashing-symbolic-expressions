# Original implementation: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
# Modified by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

from deap import tools 
from deap.benchmarks.tools import hypervolume
import numpy as np


def nsga2_deap(toolbox, NGEN, MU, verbosity,
               simplify, simplify_only_last, X, y):


    def calculate_statistics(ind):
        on_train = ind.fitness.values
        on_val   = toolbox.evaluateValidation(ind)

        return (*on_train, *on_val) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("avg", np.mean, axis=0)
    stats.register("med", np.median, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals', 'best_size', 'n_simplifications', 'n_new_hashes'] + \
                     [f"{stat} {partition} {objective}"
                         for stat in ['avg', 'med', 'std', 'min', 'max']
                         for partition in ['train', 'val']
                         for objective in toolbox.get_objectives()]

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    n_simplifications = 0
    n_new_hashes      = 0
    if simplify: # At this point, no simplification is expected, but a huge number
                 # of new hashes should be created
        pop, refit = toolbox.simplify_pop(pop, X, y, replace_pop=(not simplify_only_last))

        n_simplifications = toolbox.get_n_simplifications()
        n_new_hashes      = toolbox.get_n_new_hashes()

        # fit to get rid or uncertainty inserted by the simplify tolerance
        for idx in refit:
            pop[idx].fitness.values = toolbox.evaluate(pop[idx])

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.survive(pop, len(pop))

    # Finding the size (obj2) of the individual with best error (obj1)
    best_size = max( range(len(pop)),
        key=lambda index: ( pop[index].fitness.values[0]*pop[index].fitness.weights[0],
                            pop[index].fitness.values[1]*pop[index].fitness.weights[1]) )
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), best_size=pop[best_size].fitness.values[1],
                   n_simplifications=n_simplifications,
                   n_new_hashes=n_new_hashes,
                   **record)

    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        
        # print(f'--------- gen {gen} pop -----------')
        # for ind in pop:
        #     print(ind.fitness, ind)

        parents = toolbox.select(pop, len(pop))

        # print(f'--------- gen {gen} parents -----------')
        # for ind in parents:
        #     print(ind.fitness, ind)

        offspring = toolbox.vary_pop(parents, gen, X, y)

        # print(f'--------- gen {gen} offspring (pre fit) -----------')
        # for ind in offspring:
        #     print(ind.fitness, ind)

        # Our Variator already handles refitting after varying
        # # # fitnesses = toolbox.map(toolbox.evaluate, offspring)
        # # # for ind, fit in zip(offspring, fitnesses):
        # # #     ind.fitness.values = fit

        # simplifying offspring (alleaviate bloat by replacing based on hash)
        if simplify:
            gen_update = (gen==NGEN-1) if simplify_only_last else True

            # print(f'--------- gen {gen} -----------')
            # for ind in pop:
            #     print(ind.fitness, ind)

            offspring, refit = toolbox.simplify_pop(offspring, X, y, replace_pop=gen_update)

            n_simplifications = toolbox.get_n_simplifications()
            n_new_hashes      = toolbox.get_n_new_hashes()

            for idx in refit:
                assert offspring[idx].fitness.weights == pop[0].fitness.weights
                offspring[idx].fitness.values = toolbox.evaluate(offspring[idx])

            # print(f'--------- gen {gen} -----------')
            # for ind in pop:
            #     print(ind.fitness, ind)

        # print(f'--------- gen {gen} offspring -----------')
        # for ind in offspring:
        #     print(ind.fitness, ind)
            
        # Select the next generation population
        pop = toolbox.survive(pop + offspring, MU)

        # print(f'--------- gen {gen} survivors -----------')
        # for ind in pop:
        #     print(ind.fitness, ind)

        best_size = max( range(len(pop)),
            key=lambda index: ( pop[index].fitness.values[0]*pop[index].fitness.weights[0],
                                pop[index].fitness.values[1]*pop[index].fitness.weights[1]) )
        
        # Log and verbose
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring), best_size=pop[best_size].fitness.values[1],
                       n_simplifications=n_simplifications,
                       n_new_hashes=n_new_hashes,
                       **record)
                
        if verbosity > 0: 
            print(logbook.stream)

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook
