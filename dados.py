# dados.py
import yfinance as yf
import pandas as pd

class ColetorDados:
    def __init__(self, tickers, benchmark, start, end):

        #Inicializa o coletor de dados com os tickers e datas.
        self.tickers = tickers
        self.benchmark = benchmark
        self.start = start
        self.end = end

    def baixar_dados(self):

        #Faz o download dos preços ajustados dos ativos e do benchmark.
        data = yf.download(self.tickers + self.benchmark, start=self.start, end=self.end, group_by="ticker", auto_adjust=True)
        adj_close = pd.DataFrame()

        for t in self.tickers + self.benchmark:
            try:
                adj_close[t] = data[t]['Close']
            except:
                continue
        adj_close.dropna(axis=1, inplace=True)
        return adj_close
