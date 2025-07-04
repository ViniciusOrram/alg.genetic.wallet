import numpy as np
import pandas as pd
from datetime import timedelta
from otimizador import OtimizadorGenetico

class SimuladorCarteira:
    def __init__(self, retornos, capital, perfil_params):
        self.retornos = retornos
        self.capital = capital
        self.params = perfil_params

    def simular_rebalanceamento_trimestral(self):
        datas = self.retornos.resample('Q').first().index
        carteira_valor, historico_pesos = [], []
        valor = self.capital

        for i in range(len(datas)):
            if i == len(datas) - 1:
                intervalo = self.retornos[datas[i]:]
            else:
                intervalo = self.retornos[datas[i]:datas[i + 1] - timedelta(days=1)]

            otm = OtimizadorGenetico(intervalo, self.params['min_weight'], self.params['max_weight'], self.params['generations'])
            pesos = otm.otimizar()
            historico_pesos.append(pesos)
            retorno = (intervalo * pesos).sum(axis=1)
            valor = (1 + retorno).cumprod() * valor
            carteira_valor.append(valor)

        return pd.concat(carteira_valor), historico_pesos
