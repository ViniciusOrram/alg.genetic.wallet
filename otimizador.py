import numpy as np
import pygad

class OtimizadorGenetico:
    def __init__(self, retornos, min_weight, max_weight, generations):
        self.retornos = retornos
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.generations = generations

    def fitness_func(self, ga, solution, _):
        weights = np.clip(solution, self.min_weight, self.max_weight)
        weights /= np.sum(weights)
        ret = np.dot(self.retornos.mean(), weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.retornos.cov(), weights)))
        return ret / vol if vol > 0 else 0

    def otimizar(self):
        ga = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=20,
            sol_per_pop=60,
            num_genes=self.retornos.shape[1],
            gene_type=float,
            init_range_low=self.min_weight,
            init_range_high=self.max_weight,
            fitness_func=self.fitness_func,
            mutation_percent_genes=[5, 20],
            mutation_type="adaptive",
            crossover_type="two_points",
            parent_selection_type="rank",
            keep_parents=5,
            allow_duplicate_genes=False
        )
        ga.run()
        sol, _, _ = ga.best_solution()
        pesos = np.clip(sol, self.min_weight, self.max_weight)
        return pesos / np.sum(pesos)
