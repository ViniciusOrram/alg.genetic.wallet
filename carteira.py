# Otimização de Carteira com Algoritmo Genético baseado em dados do YFinance
# Objetivo: encontrar a combinação ideal de ativos que maximize o retorno esperado e minimize o risco

import yfinance as yf
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
import io
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("🧠 Otimizador de Carteira com Algoritmo Genético")

# ---------------- INTERFACE DE USUÁRIO ----------------
st.sidebar.header("Configurações")
tab = st.sidebar.radio("Escolha a análise", ["Configuração e Análise", "KPIs", "Gráficos", "Detalhamento e Relatório"])
perfil_risco = st.sidebar.selectbox("Perfil de Risco", ["Risco Baixo", "Risco Médio", "Risco Alto"])
num_ativos = st.sidebar.slider("Quantidade de ativos do S&P 500", 5, 50, 15)
start_date = st.sidebar.date_input("Data inicial", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Data final", value=pd.to_datetime("2024-06-25"))
initial_capital = st.sidebar.number_input("Capital inicial (USD)", value=10000)
rebalanceamento = st.sidebar.checkbox("Simular Rebalanceamento Trimestral", value=True)

# ---------------- PARÂMETROS DE RISCO ----------------
parametros_risco = {
    "Risco Baixo": {"min_weight": 0.01, "max_weight": 0.10, "generations": 200},
    "Risco Médio": {"min_weight": 0.01, "max_weight": 0.20, "generations": 300},
    "Risco Alto":  {"min_weight": 0.01, "max_weight": 0.35, "generations": 400},
}

# ---------------- DADOS ----------------
sp500_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = sp500_table['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]
selected_tickers = tickers[:num_ativos]
benchmark = ['^GSPC']

st.write("⬇️ Baixando dados...")
data = yf.download(selected_tickers + benchmark, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)
adj_close = pd.DataFrame()
for t in selected_tickers + benchmark:
    try:
        adj_close[t] = data[t]['Close']
    except:
        st.warning(f"[!] Falha em {t}")

adj_close.dropna(inplace=True)
returns = adj_close[selected_tickers].pct_change().dropna()
benchmark_data = adj_close[benchmark[0]].pct_change().dropna()

# ---------------- FUNÇÕES AUXILIARES ----------------
def portfolio_return(weights, rets):
    return np.dot(rets.mean(), weights)

def portfolio_volatility(weights, rets):
    cov_matrix = rets.cov()
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def run_optimization(returns, min_weight, max_weight, generations):
    def fitness_func(ga, solution, _):
        weights = np.clip(solution, min_weight, max_weight)
        weights /= np.sum(weights)
        ret = portfolio_return(weights, returns)
        vol = portfolio_volatility(weights, returns)
        return ret / vol if vol > 0 else 0

    ga = pygad.GA(
        num_generations=generations,
        num_parents_mating=10,
        sol_per_pop=20,
        num_genes=len(returns.columns),
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
    return weights

# ---------------- REBALANCEAMENTO TRIMESTRAL ----------------
def simular_rebalanceamento_trimestral():
    datas = returns.resample('Q').first().index
    carteira_valor = []
    valor = initial_capital
    historico_pesos = []

    for i in range(len(datas)):
        if i == len(datas) - 1:
            intervalo = returns[datas[i]:]
        else:
            intervalo = returns[datas[i]:datas[i + 1] - timedelta(days=1)]

        min_w = parametros_risco[perfil_risco]['min_weight']
        max_w = parametros_risco[perfil_risco]['max_weight']
        gen = parametros_risco[perfil_risco]['generations']
        pesos = run_optimization(intervalo, min_w, max_w, gen)
        historico_pesos.append(pesos)
        retorno_intervalo = (intervalo * pesos).sum(axis=1)
        valor = (1 + retorno_intervalo).cumprod() * valor
        carteira_valor.append(valor)

    serie_valor = pd.concat(carteira_valor)
    return serie_valor, historico_pesos

# ---------------- KPIs ----------------
def calcular_kpis(retornos):
    retorno_medio = retornos.mean()
    retorno_anual = (1 + retorno_medio) ** 12 - 1
    volatilidade_anual = retornos.std() * np.sqrt(12)
    sharpe = (retorno_anual - 0.015) / volatilidade_anual if volatilidade_anual != 0 else 0
    sortino = (retorno_anual - 0.015) / retornos[retornos < 0].std() if retornos[retornos < 0].std() != 0 else 0
    acumulado = (1 + retornos).cumprod()
    drawdown = 1 - acumulado / acumulado.cummax()
    max_drawdown = drawdown.max()
    return retorno_anual, volatilidade_anual, {
        "Retorno Anual (%)": round(retorno_anual * 100, 2),
        "Volatilidade Anual (%)": round(volatilidade_anual * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Máximo Drawdown (%)": round(max_drawdown * 100, 2)
    }

# ---------------- EXECUÇÃO PRINCIPAL ----------------
with st.spinner("⚙️ Otimizando e simulando..."):
    serie_valor, historico_pesos = simular_rebalanceamento_trimestral() if rebalanceamento else (None, None)
    pesos_finais = historico_pesos[-1] if historico_pesos else run_optimization(returns, **parametros_risco[perfil_risco])
    retorno_total = returns if not rebalanceamento else returns.loc[serie_valor.index]
    retorno_port = (retorno_total * pesos_finais).sum(axis=1)
    valor_port = serie_valor if rebalanceamento else (1 + retorno_port).cumprod() * initial_capital
    ret, vol, kpis = calcular_kpis(retorno_port)
    df_pesos = pd.DataFrame({ 'Ativo': selected_tickers, 'Peso (%)': (pesos_finais * 100).round(2) })

# ---------------- TABS ----------------
if tab == "Configuração e Análise":
    st.subheader("🎯 Pesos Otimizados")
    st.dataframe(df_pesos)
    destaque = df_pesos.sort_values(by='Peso (%)', ascending=False).head(5)
    st.markdown("**🔝 Ativos com maior peso:**")
    st.write(destaque)

elif tab == "KPIs":
    st.subheader("📊 KPIs da Carteira")
    st.json(kpis)

elif tab == "Gráficos":
    st.subheader("📈 Evolução da Carteira vs Benchmark")
    benchmark_val = (1 + benchmark_data.loc[valor_port.index]).cumprod() * initial_capital
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valor_port, label="Carteira Otimizada", linewidth=2)
    ax.plot(benchmark_val, label="S&P 500", linestyle='--')
    ax.set_title("Evolução da Carteira")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("🧮 Risco vs Retorno")
    fig2, ax2 = plt.subplots()
    ax2.scatter(vol, ret, c='green', s=100, label='Carteira')
    bm_vol = benchmark_data.std() * np.sqrt(12)
    bm_ret = (1 + benchmark_data.mean()) ** 12 - 1
    ax2.scatter(bm_vol, bm_ret, c='red', s=80, label='S&P 500')
    ax2.set_xlabel('Volatilidade Anual (Risco)')
    ax2.set_ylabel('Retorno Anual')
    ax2.set_title('Dispersão Risco x Retorno')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

elif tab == "Detalhamento e Relatório":
    st.subheader("📄 Informações da Execução")
    st.write(f"Perfil de Risco: {perfil_risco}")
    st.write(f"Período: {start_date} a {end_date}")
    st.write(f"Rebalanceamento Trimestral: {'Ativado' if rebalanceamento else 'Desativado'}")
    st.write(f"Valor final da carteira: ${valor_port[-1]:,.2f}")
    st.write("\nKPIs:")
    st.json(kpis)
