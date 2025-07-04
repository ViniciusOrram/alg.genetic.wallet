import numpy as np

class AnalisadorKPI:
    @staticmethod
    def calcular(retornos):
        r_mensal = retornos.mean()
        r_anual = (1 + r_mensal) ** 12 - 1
        vol = retornos.std() * np.sqrt(12)
        sharpe = (r_anual - 0.015) / vol if vol != 0 else 0
        sortino = (r_anual - 0.015) / retornos[retornos < 0].std() if retornos[retornos < 0].std() != 0 else 0
        acumulado = (1 + retornos).cumprod()
        dd = 1 - acumulado / acumulado.cummax()
        return r_anual, vol, {
            "Retorno Anual (%)": round(r_anual * 100, 2),
            "Volatilidade Anual (%)": round(vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Máximo Drawdown (%)": round(dd.max() * 100, 2)
        }