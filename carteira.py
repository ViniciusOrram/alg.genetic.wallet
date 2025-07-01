# Otimização de Carteira com Algoritmo Genético baseado em dados do YFinance
# Estrutura refatorada com classes para organização e eficiência

import yfinance as yf
import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta

# Configuração da interface do Streamlit
st.set_page_config(layout="wide")
st.title("Otimizador de Carteira com Algoritmo Genético")


# ---------------- Classe para Coleta de Dados ----------------
class ColetorDados:
    def __init__(self, tickers, benchmark, start, end):
        self.tickers = tickers
        self.benchmark = benchmark
        self.start = start
        self.end = end

    # Baixa dados históricos ajustados dos tickers e benchmark
    def baixar_dados(self):
        data = yf.download(self.tickers + self.benchmark, start=self.start, end=self.end, group_by="ticker",
                           auto_adjust=True)
        adj_close = pd.DataFrame()
        for t in self.tickers + self.benchmark:
            try:
                adj_close[t] = data[t]['Close']
            except:
                continue
        adj_close.dropna(axis=1, inplace=True)
        return adj_close

# ------------------- Classe para Otimização Genética -------------------
class OtimizadorGenetico:
    def __init__(self, retornos, min_weight, max_weight, generations):
        self.retornos = retornos # DataFrame com retornos dos ativos
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.generations = generations

    # Função de fitness baseada no índice de Sharpe
    def fitness_func(self, ga, solution, _):
        weights = np.clip(solution, self.min_weight, self.max_weight)
        weights /= np.sum(weights)
        ret = np.dot(self.retornos.mean(), weights) # retorno esperado
        vol = np.sqrt(np.dot(weights.T, np.dot(self.retornos.cov(), weights))) # volatilidade
        return ret / vol if vol > 0 else 0

    def otimizar(self):
        # Executa o algoritmo genético para encontrar os melhores pesos
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
            crossover_type="two_points", # compatível com carteiras pequenas
            parent_selection_type="rank",
            keep_parents=5,
            allow_duplicate_genes=False
        )
        ga.run()
        sol, _, _ = ga.best_solution()
        pesos = np.clip(sol, self.min_weight, self.max_weight)
        return pesos / np.sum(pesos)

# ------------------- Classe para Simulação com Rebalanceamento -------------------
class SimuladorCarteira:
    def __init__(self, retornos, capital, perfil_params):
        self.retornos = retornos
        self.capital = capital
        self.params = perfil_params

    # Realiza simulação de rebalanceamento da carteira a cada trimestre
    def simular_rebalanceamento_trimestral(self):
        datas = self.retornos.resample('Q').first().index
        carteira_valor, historico_pesos = [], []
        valor = self.capital

        for i in range(len(datas)):
            if i == len(datas) - 1:
                intervalo = self.retornos[datas[i]:]
            else:
                intervalo = self.retornos[datas[i]:datas[i + 1] - timedelta(days=1)]

            otm = OtimizadorGenetico(intervalo, self.params['min_weight'], self.params['max_weight'],
                                     self.params['generations'])
            pesos = otm.otimizar()
            historico_pesos.append(pesos)
            retorno = (intervalo * pesos).sum(axis=1)
            valor = (1 + retorno).cumprod() * valor
            carteira_valor.append(valor)

        return pd.concat(carteira_valor), historico_pesos

# ------------------- Classe para Cálculo de KPIs -------------------
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


# ------------------- Interface Streamlit -------------------

st.sidebar.header("Configurações")
tab = st.sidebar.radio("Escolha a análise", ["Configuração e Análise", "KPIs", "Gráficos"])
perfil_risco = st.sidebar.selectbox("Perfil de Risco", ["Risco Baixo", "Risco Médio", "Risco Alto"])
start_date = st.sidebar.date_input("Data inicial", value=pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("Data final", value=pd.to_datetime("2025-07-01"))
capital = st.sidebar.number_input("Capital inicial (USD)", value=100)
rebalancear = st.sidebar.checkbox("Simular Rebalanceamento Trimestral", value=False)

# Parâmetros de otimização por perfil
parametros = {
    "Risco Baixo": {"min_weight": 0.01, "max_weight": 0.10, "generations": 200, "ativos": 10},
    "Risco Médio": {"min_weight": 0.01, "max_weight": 0.20, "generations": 300, "ativos": 5},
    "Risco Alto": {"min_weight": 0.01, "max_weight": 0.35, "generations": 400, "ativos": 2}
}

perfil = parametros[perfil_risco]
# Coleta dos dados
tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].str.replace('.', '-',                                                                                                           regex=False).tolist()
coletor = ColetorDados(tickers, ['^GSPC'], start_date, end_date)
st.write("Baixando dados...")
adj_close = coletor.baixar_dados()
returns_full = adj_close.pct_change().dropna()

# Seleção dos melhores ativos com base no retorno médio
melhores = returns_full.mean().sort_values(ascending=False).head(perfil['ativos'])
retornos = returns_full[melhores.index]

with st.spinner("Otimizando e simulando..."):
    if rebalancear:
        sim = SimuladorCarteira(retornos, capital, perfil)
        valor_port, pesos_hist = sim.simular_rebalanceamento_trimestral()
        pesos_finais = pesos_hist[-1]
    else:
        otimizador = OtimizadorGenetico(retornos, perfil['min_weight'], perfil['max_weight'], perfil['generations'])
        pesos_finais = otimizador.otimizar()
        retorno_port = (retornos * pesos_finais).sum(axis=1)
        valor_port = capital * (1 + retorno_port).cumprod()

# Cálculo dos KPIs
retorno_port = (retornos * pesos_finais).sum(axis=1)
r_anual, vol, kpis = AnalisadorKPI.calcular(retorno_port)
df_pesos = pd.DataFrame({"Ativo": melhores.index, "Peso (%)": (pesos_finais * 100).round(2)})

# ---------------- Exibição ----------------

# Comparativo entre carteiras por perfil de risco
st.subheader("Comparativo entre Perfis de Risco")
comparativos = {}
for nome, param in parametros.items():
    ativos_perf = returns_full.mean().sort_values(ascending=False).head(param['ativos']).index
    retornos_perf = returns_full[ativos_perf]
    otm = OtimizadorGenetico(retornos_perf, param['min_weight'], param['max_weight'], param['generations'])
    pesos_perf = otm.otimizar()
    ret_perf = (retornos_perf * pesos_perf).sum(axis=1)
    val_perf = capital * (1 + ret_perf).cumprod()
    comparativos[nome] = val_perf

# Gráfico comparativo entre perfis
fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
for nome, serie in comparativos.items():
    ax_comp.plot(serie, label=nome)
ax_comp.set_title("Evolução da Carteira por Perfil de Risco")
ax_comp.set_ylabel("Valor (USD)")
ax_comp.legend()
ax_comp.grid()
st.pyplot(fig_comp)

# Análise de setores da carteira final
st.subheader("Composição Setorial da Carteira")
tabela_sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
mapa_setores = dict(zip(tabela_sp500['Symbol'].str.replace('.', '-', regex=False), tabela_sp500['GICS Sector']))
df_pesos['Setor'] = df_pesos['Ativo'].map(mapa_setores)
df_setor = df_pesos.groupby('Setor')['Peso (%)'].sum().sort_values(ascending=False).reset_index()
st.dataframe(df_setor)
fig_setor, ax_setor = plt.subplots(figsize=(10, 5))
ax_setor.bar(df_setor['Setor'], df_setor['Peso (%)'], color='salmon')
ax_setor.set_title("Distribuição por Setor")
ax_setor.set_xticklabels(df_setor['Setor'], rotation=45, ha='right')
st.pyplot(fig_setor)

# Gráfico comparativo com rebalanceamento
st.subheader("Rentabilidade: Rebalanceado vs Fixo")
if rebalancear:
    otm_fixo = OtimizadorGenetico(retornos, perfil['min_weight'], perfil['max_weight'], perfil['generations'])
    pesos_fixos = otm_fixo.otimizar()
    ret_fixo = (retornos * pesos_fixos).sum(axis=1)
    val_fixo = capital * (1 + ret_fixo).cumprod()
    fig_rb, ax_rb = plt.subplots(figsize=(12, 5))
    ax_rb.plot(valor_port, label='Rebalanceado', linewidth=2)
    ax_rb.plot(val_fixo, label='Fixo', linestyle='--')
    ax_rb.set_title("Rentabilidade Acumulada")
    ax_rb.set_ylabel("Valor (USD)")
    ax_rb.legend()
    ax_rb.grid()
    st.pyplot(fig_rb)

# Análise de Sensibilidade: variação de 5% nos pesos
st.subheader("Análise de Sensibilidade")
variacoes = [-0.05, 0.05]
sensibilidade = []
for i in range(len(pesos_finais)):
    for delta in variacoes:
        pesos_perturb = np.copy(pesos_finais)
        pesos_perturb[i] = np.clip(pesos_perturb[i] + delta, perfil['min_weight'], perfil['max_weight'])
        pesos_perturb /= np.sum(pesos_perturb)
        ret_sim = (retornos * pesos_perturb).sum(axis=1)
        _, _, kpi_sim = AnalisadorKPI.calcular(ret_sim)
        sensibilidade.append({"Ativo": melhores.index[i], "Variação": f"{int(delta * 100)}%", **kpi_sim})

st.dataframe(pd.DataFrame(sensibilidade))

# Exibição de conteúdo conforme aba selecionada
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
    st.subheader("Evolução da Carteira")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(valor_port, label="Carteira Otimizada", linewidth=2)
    ax.set_title("Evolução da Carteira")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (USD)")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

    st.subheader("Risco vs Retorno")
    fig2, ax2 = plt.subplots()
    ax2.scatter(vol, r_anual, c='green', s=100, label='Carteira')
    ax2.set_xlabel('Volatilidade Anual (Risco)')
    ax2.set_ylabel('Retorno Anual')
    ax2.set_title('Dispersão Risco x Retorno')
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
