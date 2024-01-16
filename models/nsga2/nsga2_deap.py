# Original implementation: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
# Modified by Guilherme Aldeia 01/08/2024 guilherme.aldeia@ufabc.edu.br

from deap import tools 
from deap.benchmarks.tools import hypervolume
import numpy as np


def nsga2_deap(toolbox, NGEN, MU, CXPB, verbosity, random, 
               simplify, simplify_only_last, X, y):

    # NGEN = 250
    # MU = 100
    # CXPB = 0.9

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
        pop = toolbox.simplify_pop(pop, X, y, replace_pop=(not simplify_only_last))

        n_simplifications = toolbox.get_n_simplifications()
        n_new_hashes      = toolbox.get_n_new_hashes()

        if not simplify_only_last:
            # fit to get rid or uncertainty inserted by the simplify tolerance
            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

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
        parents = toolbox.select(pop, len(pop))

        offspring = []
        for ind1, ind2 in zip(parents[::2], parents[1::2]):
            ind1 = toolbox.clone(ind1)
            ind2 = toolbox.clone(ind2)

            off1, off2 = None, None
            if random.random() < CXPB:
                off1, off2 = toolbox.mate(ind1, ind2)
            else:
                # mutate returns a list with 1 individual
                off1, = toolbox.mutate(ind1)
                off2, = toolbox.mutate(ind2)

            offspring.extend([off1])
            offspring.extend([off2])

        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # simplifying offspring (alleaviate bloat by replacing based on hash)
        n_simplifications = 0
        n_new_hashes      = 0
        if simplify:
            gen_update = (gen==NGEN-1) if simplify_only_last else True

            offspring = toolbox.simplify_pop(offspring, X, y, replace_pop=gen_update)

            n_simplifications = toolbox.get_n_simplifications()
            n_new_hashes      = toolbox.get_n_new_hashes()

            if gen_update:
                # fit to get rid or uncertainty inserted by the simplify tolerance
                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.survive(pop + offspring, MU)

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
