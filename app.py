# app.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from dados import ColetorDados
from otimizador import OtimizadorGenetico
from simulador import SimuladorCarteira
from kpi import AnalisadorKPI

# Configurações iniciais do Streamlit
st.set_page_config(layout="wide")
st.title("Otimizador de Carteira com Algoritmo Genético")

# ---------------------- Interface ----------------------
st.sidebar.header("Configurações")
tab = st.sidebar.radio("Escolha a análise", ["Configuração e Análise", "KPIs", "Gráficos"])
perfil_risco = st.sidebar.selectbox("Perfil de Risco", ["Risco Baixo", "Risco Médio", "Risco Alto"])
capital = st.sidebar.number_input("Capital inicial (USD)", value=100)
rebalancear = st.sidebar.checkbox("Simular Rebalanceamento Trimestral", value=False)

# Parâmetros por perfil
parametros = {
    "Risco Baixo": {"min_weight": 0.01, "max_weight": 0.10, "generations": 50, "ativos": 10},
    "Risco Médio": {"min_weight": 0.01, "max_weight": 0.20, "generations": 100, "ativos": 5},
    "Risco Alto": {"min_weight": 0.01, "max_weight": 0.35, "generations": 150, "ativos": 2}
}

perfil = parametros[perfil_risco]

# ---------------------- Coleta e Pré-processamento ----------------------
# Intervalo automático de 6 meses
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
coletor = ColetorDados(tickers, ['^GSPC'], start_date, end_date)

st.write("Baixando dados...")
adj_close = coletor.baixar_dados()
returns_full = adj_close.pct_change().dropna()

# Seleção dos melhores ativos
melhores = returns_full.mean().sort_values(ascending=False).head(perfil['ativos'])
retornos = returns_full[melhores.index]

# ---------------------- Otimização / Simulação ----------------------
with st.spinner("Otimizando e simulando..."):
    historico_fitness = None
    if rebalancear:
        sim = SimuladorCarteira(retornos, capital, perfil)
        valor_port, pesos_hist = sim.simular_rebalanceamento_trimestral()
        pesos_finais = pesos_hist[-1]
    else:
        otimizador = OtimizadorGenetico(retornos, perfil['min_weight'], perfil['max_weight'], perfil['generations'], early_stop_rounds=20)
        pesos_finais = otimizador.otimizar()
        historico_fitness = otimizador.historico_fitness  # capturado para exibição
        retorno_port = (retornos * pesos_finais).sum(axis=1)
        valor_port = capital * (1 + retorno_port).cumprod()

# ---------------------- KPIs ----------------------
retorno_port = (retornos * pesos_finais).sum(axis=1)
r_anual, vol, kpis = AnalisadorKPI.calcular(retorno_port)
df_pesos = pd.DataFrame({"Ativo": melhores.index, "Peso (%)": (pesos_finais * 100).round(2)})

# ---------------------- Tabs ----------------------
if tab == "Configuração e Análise":
    st.subheader("Pesos Otimizados")
    st.dataframe(df_pesos)
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    ax_bar.bar(df_pesos['Ativo'], df_pesos['Peso (%)'], color='skyblue')
    ax_bar.set_ylabel("Proporção (%)")
    ax_bar.set_xlabel("Ações")
    ax_bar.set_title("Distribuição da Carteira por Ativo")
    ax_bar.tick_params(axis='x', rotation=45)
    st.pyplot(fig_bar)

elif tab == "KPIs":
    st.subheader("KPIs da Carteira")
    st.json(kpis)

elif tab == "Gráficos":
    # Comparativo entre perfis
    st.subheader("Comparativo entre Perfis de Risco")
    comparativos = {}
    for nome, param in parametros.items():
        ativos_perf = returns_full.mean().sort_values(ascending=False).head(param['ativos']).index
        retornos_perf = returns_full[ativos_perf]
        otm = OtimizadorGenetico(retornos_perf, param['min_weight'], param['max_weight'], param['generations'], early_stop_rounds=20)
        pesos_perf = otm.otimizar()
        ret_perf = (retornos_perf * pesos_perf).sum(axis=1)
        val_perf = capital * (1 + ret_perf).cumprod()
        comparativos[nome] = val_perf

    fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
    for nome, serie in comparativos.items():
        ax_comp.plot(serie, label=nome)
    ax_comp.set_title("Evolução da Carteira por Perfil de Risco")
    ax_comp.set_ylabel("Valor (USD)")
    ax_comp.legend()
    ax_comp.grid()
    st.pyplot(fig_comp)

    # Gráfico comparativo com benchmark S&P 500
    st.subheader("Carteira vs S&P 500")
    sp500 = adj_close['^GSPC']
    sp500_ret = sp500.pct_change().dropna()
    sp500_val = capital * (1 + sp500_ret).cumprod()

    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 6))
    ax_cmp.plot(valor_port, label="Carteira", linewidth=2)
    ax_cmp.plot(sp500_val, label="S&P 500", linestyle="--")
    ax_cmp.set_title("Comparativo com S&P 500")
    ax_cmp.set_ylabel("Valor (USD)")
    ax_cmp.legend()
    ax_cmp.grid()
    st.pyplot(fig_cmp)

    # Dispersão Risco x Retorno
    st.subheader("Risco vs Retorno")
    fig2, ax2 = plt.subplots()
    ax2.scatter(vol, r_anual, c='green', s=100, label='Carteira')
    ax2.set_xlabel('Volatilidade Anual (Risco)')
    ax2.set_ylabel('Retorno Anual')
    ax2.set_title('Dispersão Risco x Retorno')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Gráficos de Evolução do Fitness do Algoritmo Genético
    st.subheader("Evolução do Algoritmo Genético")

    # Fitness da melhor solução por geração
    fig_fit, ax_fit = plt.subplots()
    ax_fit.plot(historico_fitness["melhor"], color="blue", label="Melhor Fitness")
    ax_fit.plot(historico_fitness["media"], color="orange", linestyle="--", label="Fitness Médio")
    ax_fit.set_title("Evolução do Índice de Fitness por Geração")
    ax_fit.set_xlabel("Geração")
    ax_fit.set_ylabel("Fitness (Sharpe Ratio)")
    ax_fit.grid(True)
    ax_fit.legend()
    st.pyplot(fig_fit)
