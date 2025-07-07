import numpy as np
import pygad

#retornos: DataFrame com os retornos dos ativos (linhas: datas, colunas: ativos).
#min_weight e max_weight: limites inferior e superior de alocação de capital por ativo.
#generations: número de gerações do algoritmo genético (quantas vezes a população será evoluída).

class OtimizadorGenetico:
    def __init__(self, retornos, min_weight, max_weight, generations):
        self.retornos = retornos
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.generations = generations

    def fitness_func(self, ga, solution, _): #solution é um vetor de pesos sugeridos por um indivíduo da população.
        weights = np.clip(solution, self.min_weight, self.max_weight) #np.clip limita os valores entre min_weight e max_weight para manter a solução válida.
        weights /= np.sum(weights) #Normaliza os pesos para garantir que a soma seja 1 (ou 100%).
        ret = np.dot(self.retornos.mean(), weights) #retornos.mean() retorna o retorno médio de cada ativo
        vol = np.sqrt(np.dot(weights.T, np.dot(self.retornos.cov(), weights))) #Multiplicação matricial com os pesos para calcular o desvio padrão total da carteira.
        return ret / vol if vol > 0 else 0 #Retorna o índice de Sharpe (retorno / risco).

    def otimizar(self):
        ga = pygad.GA(
            num_generations=self.generations, #Número de gerações para evoluir a população.
            num_parents_mating=20, #Quantos indivíduos serão selecionados como pais por geração.
            sol_per_pop=60, #Tamanho da população (quantas soluções existem por geração).
            num_genes=self.retornos.shape[1], #Número de genes = número de ativos (cada gene é o peso de um ativo).
            gene_type=float, #Os genes (pesos) são números do tipo float.
            init_range_low=self.min_weight,
            init_range_high=self.max_weight, #Intervalo inicial dos valores dos genes (pesos iniciais).
            fitness_func=self.fitness_func, #A função de avaliação que será usada para pontuar cada solução.
            #Mutação adaptativa: varia aleatoriamente de 5% a 20% dos genes por indivíduo.
            #Ajuda a manter diversidade na população e escapar de mínimos locais.
            mutation_percent_genes=[5, 20],
            mutation_type="adaptive",
            crossover_type="two_points",#Usa crossover de dois pontos, técnica de recombinação genética para criar filhos.
            parent_selection_type="rank", #Seleção por ranking: os melhores indivíduos têm maior chance de serem escolhidos como pais.
            keep_parents=5, #Preserva os 5 melhores indivíduos entre gerações.
            #Não permite que genes repetidos existam em uma mesma solução (mais útil em problemas com restrições, mas aqui evita concentrações excessivas).
            allow_duplicate_genes=False
        )
        ga.run()
        sol, _, _ = ga.best_solution() #Obtém o melhor vetor de pesos (solução) encontrado após todas as gerações.
        #Garante que os pesos estejam dentro dos limites válidos e normaliza para somar 1.
        pesos = np.clip(sol, self.min_weight, self.max_weight)
        return pesos / np.sum(pesos)
