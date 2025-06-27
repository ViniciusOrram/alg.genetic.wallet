# Otimização de Carteira com Algoritmo Genético baseado em dados do YFinance
# Objetivo: encontrar a combinação ideal de ativos que maximize o retorno esperado e minimize o risco

import yfinance as yf
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt

# ---------------- CONFIGURAÇÕES ----------------
num_ativos = 15
benchmark = ['^GSPC']  # S&P 500
start_date = "2022-01-01"
end_date = "2024-06-25"
risk_free_rate = 0.015
min_weight = 0.01
max_weight = 0.20
initial_capital = 10000

# ---------------- SELEÇÃO DOS ATIVOS DO S&P 500 ----------------
print("🔍 Coletando tickers do S&P 500...")
sp500_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = sp500_table['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]  # Ex: BRK.B -> BRK-B
selected_tickers = tickers[:num_ativos]

# ---------------- DOWNLOAD DE DADOS ----------------
print("⬇️ Baixando dados dos ativos...")
data = yf.download(selected_tickers + benchmark, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)

adj_close = pd.DataFrame()
for t in selected_tickers + benchmark:
    try:
        adj_close[t] = data[t]['Close']
    except:
        print(f"[!] Falha em {t}")

adj_close.dropna(inplace=True)
returns = adj_close[selected_tickers].pct_change().dropna()

# ---------------- FUNÇÃO OBJETIVO ----------------
def portfolio_return(weights):
    return np.dot(returns.mean(), weights)

def portfolio_volatility(weights):
    cov_matrix = returns.cov()
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def fitness_func(ga, solution, _):
    weights = np.clip(solution, min_weight, max_weight)
    weights /= np.sum(weights)
    ret = portfolio_return(weights)
    vol = portfolio_volatility(weights)
    return ret / vol if vol > 0 else 0

# ---------------- ALGORITMO GENÉTICO ----------------
print("⚙️ Rodando Algoritmo Genético...")
ga = pygad.GA(
    num_generations=300,
    num_parents_mating=10,
    sol_per_pop=20,
    num_genes=len(selected_tickers),
    gene_type=float,
    init_range_low=min_weight,
    init_range_high=max_weight,
    fitness_func=fitness_func,
    mutation_percent_genes=20,
    mutation_type="random",
    crossover_type="single_point",
    allow_duplicate_genes=False
)

ga.run()

solution, _, _ = ga.best_solution()
weights = np.clip(solution, min_weight, max_weight)
weights /= np.sum(weights)

# ---------------- BACKTEST ----------------
monthly_returns = returns.resample('M').apply(lambda x: (x + 1).prod() - 1)
portfolio_returns = (monthly_returns * weights).sum(axis=1)
portfolio_value = (1 + portfolio_returns).cumprod() * initial_capital

benchmark_data = adj_close[benchmark[0]].pct_change().resample('M').apply(lambda x: (x + 1).prod() - 1)
benchmark_value = (1 + benchmark_data).cumprod() * initial_capital

# ---------------- GRÁFICO ----------------
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value, label="Carteira Otimizada", linewidth=2)
plt.plot(benchmark_value, label="S&P 500", linestyle='--')
plt.title("Evolução da Carteira vs Benchmark")
plt.xlabel("Data")
plt.ylabel("Valor (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---------------- RESULTADOS ----------------
print("\n🎯 Pesos Otimizados:")
for t, w in zip(selected_tickers, weights):
    print(f"{t}: {w:.2%}")
