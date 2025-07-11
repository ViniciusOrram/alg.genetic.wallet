import numpy as np
import pygad

class OtimizadorGenetico:
    def __init__(self, retornos, min_weight, max_weight, generations, early_stop_rounds=20):
        self.retornos = retornos
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.generations = generations
        self.early_stop_rounds = early_stop_rounds
        self.historico_fitness = {"melhor": [], "media": []}

    def fitness_func(self, ga, solution, _):
        weights = np.clip(solution, self.min_weight, self.max_weight)
        weights /= np.sum(weights)
        retorno_esperado = np.dot(self.retornos.mean(), weights)
        volatilidade = np.sqrt(np.dot(weights.T, np.dot(self.retornos.cov(), weights)))
        sharpe = retorno_esperado / volatilidade if volatilidade > 0 else 0

        print(f"[Fitness] Weights: {weights.round(3)}, Sharpe: {sharpe:.4f}")
        return sharpe

    def _callback(self, ga_instance):
        # Fitness da melhor e média da geração atual
        best = np.max(ga_instance.last_generation_fitness)
        mean = np.mean(ga_instance.last_generation_fitness)

        self.historico_fitness["melhor"].append(best)
        self.historico_fitness["media"].append(mean)

        if len(self.historico_fitness["melhor"]) >= self.early_stop_rounds:
            recentes = self.historico_fitness["melhor"][-self.early_stop_rounds:]
            if max(recentes) - min(recentes) < 1e-6:
                print(f"[Early Stopping] Estabilizado nas últimas {self.early_stop_rounds} gerações. Encerrando.")
                ga_instance.running = False
                ga_instance.running = False

    def otimizar(self):

        def log_progresso(ga):
            best_solution, best_fitness, _ = ga.best_solution()
            print(f"[Geração {ga.generations_completed}] Melhor Sharpe: {best_fitness:.4f}")
            self._callback(ga)  # ← Chama o callback a cada geração

        print("[Início da Otimização Genética]")

        ga = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=20,
            sol_per_pop=60,
            num_genes=self.retornos.shape[1],
            gene_type=float,
            init_range_low=self.min_weight,
            init_range_high=self.max_weight,
            fitness_func=self.fitness_func,
            on_generation=log_progresso,  # ← callback chamado aqui
            mutation_percent_genes=[5, 20],
            mutation_type="adaptive",
            crossover_type="two_points",
            parent_selection_type="rank",
            keep_parents=5,
            allow_duplicate_genes=False
        )

        ga.run()
        print("[Fim da Otimização Genética]")
        self.geracoes_efetivas = ga.generations_completed

        best_solution, best_fitness, _ = ga.best_solution()
        pesos = np.clip(best_solution, self.min_weight, self.max_weight)
        pesos /= np.sum(pesos)

        print(f"[Resultado Final] Pesos Otimizados: {pesos.round(3)}")
        print(f"[Resultado Final] Sharpe: {best_fitness:.4f}")

        return pesos
