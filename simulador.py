import numpy as np
import pandas as pd
from datetime import timedelta
from otimizador import OtimizadorGenetico

class SimuladorCarteira:
    def __init__(self, retornos, capital, perfil_params):
        """
        Simula carteira com rebalanceamento trimestral.
        - retornos: matriz de retornos
        - capital: valor inicial investido
        - perfil_params: dicionário com configuração do perfil de risco
        """
        self.retornos = retornos
        self.capital = capital
        self.params = perfil_params

    def simular_rebalanceamento_trimestral(self, early_stop_rounds=20):
        """
        Para cada trimestre, reotimiza os pesos e acumula o valor da carteira.
        Retorna o histórico de valor da carteira e os pesos usados a cada trimestre.
        """
        datas = self.retornos.resample('Q').first().index
        carteira_valor, historico_pesos, historico_fitness = [], [], []
        valor = self.capital

        for i in range(len(datas)):
            if i == len(datas) - 1:
                intervalo = self.retornos[datas[i]:]
            else:
                intervalo = self.retornos[datas[i]:datas[i + 1] - timedelta(days=1)]

            otm = OtimizadorGenetico(
                intervalo,
                self.params['min_weight'],
                self.params['max_weight'],
                self.params['generations'],
                early_stop_rounds=early_stop_rounds  # <-- early stopping aplicado
            )
            pesos = otm.otimizar()
            historico_pesos.append(pesos)
            historico_fitness.append(otm.historico_fitness)  # <-- adiciona histórico por trimestre

            # Retorno ponderado da carteira
            retorno_carteira = (intervalo * pesos).sum(axis=1)
            #Alteração calculo campo valor
            valor = (1 + retorno_carteira).cumprod() * valor.iloc[-1] if isinstance(valor, pd.Series) else (1 + retorno_carteira).cumprod() * valor
            carteira_valor.append(valor)

        return pd.concat(carteira_valor), historico_pesos, historico_fitness
