# -*- coding: utf-8 -*-

#Import cell
from google.colab import files
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import holidays as hdays
import matplotlib.ticker as mticker
from datetime import timedelta
from datetime import datetime
import calendar
import seaborn as sns
import requests
from matplotlib.ticker import FuncFormatter
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler

"""# Data Cleaning

"""

#Data Cleaning and Organization
drive.mount('/content/drive')

df_consultas = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/PROJETO HUMANITAR/consultas_integral.csv",sep=";")
df_estrutura = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/PROJETO HUMANITAR/estrutura.csv",sep=";")

#df_consultas = pd.read_csv("/content/drive/MyDrive/ColabNotebooks/andre/humanitar/PROJETO HUMANITAR/consultas_integral.csv",sep=";")
#df_estrutura = pd.read_csv("/content/drive/MyDrive/ColabNotebooks/andre/humanitar/PROJETO HUMANITAR/estrutura.csv",sep=";")

df_merged = pd.merge(df_consultas, df_estrutura, on='idservico', how='left')

#Roubei do bi
df_merged['hora'] = pd.to_datetime(df_merged['hora'], format='%H:%M').dt.hour
df_merged['data'] = pd.to_datetime(df_merged['data'], format='%d/%m/%Y')

hoje = pd.Timestamp(datetime.today().date())
df_merged = df_merged[df_merged['data'] <= hoje]

df_merged = df_merged.sort_values(['idservico', 'data', 'hora']).reset_index(drop=True)

hoje = pd.Timestamp.today().normalize()
df_merged = df_merged[df_merged['data'] <= hoje]

df_prev = df_merged.copy()
df_next = df_merged.copy()

df_prev['data'] = df_prev['data'] + timedelta(days=7)
df_next['data'] = df_next['data'] - timedelta(days=7)

df_prev = df_prev.rename(columns={'consultas': 'consultas_prev'})
df_next = df_next.rename(columns={'consultas': 'consultas_next'})

df_clean = df_merged.merge(
    df_prev[['idservico', 'hora', 'data', 'consultas_prev']],
    on=['idservico', 'hora', 'data'], how='left'
)
df_clean = df_clean.merge(
    df_next[['idservico', 'hora', 'data', 'consultas_next']],
    on=['idservico', 'hora', 'data'], how='left'
)

df_clean['media_ref'] = df_clean[['consultas_prev', 'consultas_next']].mean(axis=1)

mask = (df_clean['consultas'] > 2 * df_clean['media_ref']) & (df_clean['consultas'] > 10)
df_clean.loc[mask, 'consultas'] = df_clean.loc[mask, 'media_ref']

df_clean = df_clean.drop(columns=['consultas_prev', 'consultas_next', 'media_ref'])
df_consultas = df_clean

"""# Data Analysis

## 1.0 - Média de consultas

Por Mês
"""

plt.style.use("seaborn-v0_8-whitegrid")

#Criando colunas com o nome e numero dos meses
df_consultas['mes_num'] = df_consultas['data'].dt.month
df_consultas['mes_nome'] = df_consultas['data'].dt.month_name()

#Bi que fez pra deixar os meses em ordem
meses_ordem = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

traducao_meses = {
    'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março',
    'April': 'Abril', 'May': 'Maio', 'June': 'Junho',
    'July': 'Julho', 'August': 'Agosto', 'September': 'Setembro',
    'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
}

df_consultas['mes_nome'] = df_consultas['mes_nome'].map(traducao_meses)
meses_ordem_pt = list(traducao_meses.values())


#Calculando a media por mês
monthly = df_consultas.groupby(['mes_nome'])['consultas'].mean().reindex(meses_ordem_pt)


#Tema e tipografia
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120,"axes.titlesize": 18,"axes.labelsize": 13,"xtick.labelsize": 11,"ytick.labelsize": 11,})

#Cores
COR = "#4E79A7"
COR_SET = "#F28E2B"
COR_HOSP = "#59A14F"

# ---------- GERAL ----------
fig, ax = plt.subplots(figsize=(12,5))
ax.bar(monthly.index, monthly, width=0.65, color=COR, edgecolor="black", linewidth=0.6)

ax.set_title("Média de Consultas por Mês")
ax.set_xlabel("Mês"); ax.set_ylabel("Número de Consultas")
ax.ticklabel_format(style="plain", axis="y")
ax.set_axisbelow(True)

#Linha da média anual
ax.axhline(monthly.mean(), ls="--", lw=1.5, color="#888", label="Média anual")

ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.show()

# ---------- POR SETOR ----------
for setor, df_setor in df_consultas.groupby("setor"):
    montly_setor = (df_setor.groupby(["mes_nome"])["consultas"].mean().reindex(meses_ordem_pt)) #Calculando medias por setor e deixando os meses em ordem

    #Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(montly_setor.index, montly_setor, width=0.65,color=COR_SET, edgecolor="black", linewidth=0.6)
    ax.set_title(f"Média de Consultas por Mês — Setor: {setor}")
    ax.set_xlabel("Mês"); ax.set_ylabel("Número de Consultas")
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_axisbelow(True)
    ax.axhline(montly_setor.mean(), ls="--", lw=1.5, color="#888", label="Média do setor")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()

# ---------- POR HOSPITAL ----------
for hosp, df_hosp in df_consultas.groupby("cliente"):
    montly_hosp = (df_hosp.groupby(["mes_nome"])["consultas"].mean().reindex(meses_ordem_pt)) #Calculando medias por hospital e deixando os meses em ordem

    #Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(montly_hosp.index, montly_hosp, width=0.65,color=COR_HOSP, edgecolor="black", linewidth=0.6)
    ax.set_title(f"Média de Consultas por Mês — Hospital: {hosp}")
    ax.set_xlabel("Mês"); ax.set_ylabel("Número de Consultas")
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_axisbelow(True)
    ax.axhline(montly_hosp.mean(), ls="--", lw=1.5, color="#888", label="Média do hospital")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.show()

"""Por horas do dia"""

#Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi":120, "axes.titlesize":18, "axes.labelsize":13,"xtick.labelsize":11, "ytick.labelsize":11})


#Cores
COR_GERAL = "#4E79A7"
COR_SETOR = "#F28E2B"
COR_HOSP  = "#59A14F"

# --------- GERAL ---------

#Calculando media
hourly = (df_consultas.groupby("hora")["consultas"].mean().reindex(range(24)))

#Plot
fig, ax = plt.subplots(figsize=(12,5))
ax.bar(hourly.index, hourly.values, width=0.65, color=COR_GERAL, edgecolor="black", linewidth=0.5)
ax.set_title("Média de Consultas por Hora do Dia — Geral")
ax.set_xlabel("Hora do Dia"); ax.set_ylabel("Número de Consultas")
ax.tick_params(axis="x", rotation=0); ax.ticklabel_format(style="plain", axis="y")
ax.axhline(hourly.mean(), ls="--", lw=1.5, color="#888", label="Média")
h_peak = int(hourly.idxmax()); y_peak = float(hourly.max())
ax.plot([h_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {h_peak}h")
ax.text(h_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout(); plt.show()

# --------- POR SETOR ---------
for setor, df_set in df_consultas.groupby("setor"):
    hourly_setor = (df_set.groupby("hora")["consultas"].mean().reindex(range(24))) #Calculando media

    #Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(hourly_setor.index, hourly_setor.values, width=0.65,color=COR_SETOR, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Hora — Setor: {setor}")
    ax.set_xlabel("Horário"); ax.set_ylabel("Número de Consultas")
    ax.tick_params(axis="x", rotation=0); ax.ticklabel_format(style="plain", axis="y")
    ax.axhline(hourly_setor.mean(), ls="--", lw=1.5, color="#888", label="Média")
    h_peak = int(hourly_setor.idxmax()); y_peak = float(hourly_setor.max())
    ax.plot([h_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {h_peak}h")
    ax.text(h_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

# --------- POR HOSPITAL ---------
for hosp, df_h in df_consultas.groupby("cliente"):
    hourly_hosp = (df_h.groupby("hora")["consultas"].mean().reindex(range(24)))

    #Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(hourly_hosp.index, hourly_hosp.values, width=0.65,color=COR_HOSP, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Hora — Hospital: {hosp}")
    ax.set_xlabel("Horário"); ax.set_ylabel("Número de Consultas")
    ax.tick_params(axis="x", rotation=0); ax.ticklabel_format(style="plain", axis="y")
    ax.axhline(hourly_hosp.mean(), ls="--", lw=1.5, color="#888", label="Média")
    h_peak = int(hourly_hosp.idxmax()); y_peak = float(hourly_hosp.max())
    ax.plot([h_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {h_peak}h")
    ax.text(h_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

"""Por dias da semana"""

#Coluna com os nomes dos dias da semana
df_consultas["dia"] = df_consultas["data"].dt.day_name()

# Tema e tipografia
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120,"axes.titlesize": 18,"axes.labelsize": 13,"xtick.labelsize": 11,"ytick.labelsize": 11,})

#Cores
COR_GERAL = "#4E79A7"
COR_SETOR = "#F28E2B"
COR_HOSP  = "#59A14F"

#Ordem dos dias
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# --------- GERAL ---------

#Calculando a media e deixando os dias em ordem
daily = df_consultas.groupby("dia")["consultas"].mean().reindex(order)

#Plot
fig3, ax3 = plt.subplots(figsize=(12,5))
ax3.bar(daily.index, daily.values, width=0.65, color=COR_GERAL, edgecolor="black", linewidth=0.5)
ax3.set_title("Média de Consultas por Dia da Semana — Geral")
ax3.set_xlabel("Dia da Semana"); ax3.set_ylabel("Número de Consultas")
ax3.tick_params(axis="x", rotation=0)
ax3.ticklabel_format(style="plain", axis="y")
ax3.set_axisbelow(True)

#Linha da média e pico
ax3.axhline(daily.mean(), ls="--", lw=1.5, color="#888", label="Média")
d_peak = daily.idxmax(); y_peak = float(daily.max())
ax3.plot([d_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {d_peak}")
ax3.text(d_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)
ax3.legend(frameon=False, loc="upper left")

plt.tight_layout(); plt.show()

# --------- POR SETOR ---------
for setor, df_setor in df_consultas.groupby("setor"):
    daily_setor = (df_setor.groupby("dia")["consultas"].mean().reindex(order)) # Calculando a media de deixando os dias em ordem

    #Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(daily_setor.index, daily_setor.values, width=0.65,color=COR_SETOR, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Dia da Semana — Setor: {setor}")
    ax.set_xlabel("Dia da Semana"); ax.set_ylabel("Número de Consultas")
    ax.tick_params(axis="x", rotation=0)
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_axisbelow(True)

    #Linha da média e pico
    ax.axhline(daily_setor.mean(), ls="--", lw=1.5, color="#888", label="Média")
    d_peak = daily_setor.idxmax(); y_peak = float(daily_setor.max())
    ax.plot([d_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {d_peak}")
    ax.text(d_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

# --------- POR HOSPITAL ---------
for hosp, df_hosp in df_consultas.groupby("cliente"):
    daily_hosp = (df_hosp.groupby("dia")["consultas"].mean().reindex(order)) # Calculando a media e deixando os dias em ordem

    #Plot
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(daily_hosp.index, daily_hosp.values, width=0.65,color=COR_HOSP, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Dia da Semana — Hospital: {hosp}")
    ax.set_xlabel("Dia da Semana"); ax.set_ylabel("Número de Consultas")
    ax.tick_params(axis="x", rotation=0)
    ax.ticklabel_format(style="plain", axis="y")
    ax.set_axisbelow(True)

    #Linha da média e pico
    ax.axhline(daily_hosp.mean(), ls="--", lw=1.5, color="#888", label="Média")
    d_peak = daily_hosp.idxmax(); y_peak = float(daily_hosp.max())
    ax.plot([d_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {d_peak}")
    ax.text(d_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

"""Por feriado"""

#Criando DF nova
df = df_consultas.copy()

#Chegando type da data
df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
years = range(df["data"].dt.year.min(), df["data"].dt.year.max() + 1)

#Criando coluna dos feriados
br_hols = hdays.Brazil(years=years)
df["feriado?"] = df["data"].dt.date.map(lambda d: d in br_hols)
df["nome_feriado"] = df["data"].dt.date.map(lambda d: br_hols.get(d))

#Tema e tipografia
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120,"axes.titlesize": 18,"axes.labelsize": 13,"xtick.labelsize": 11,"ytick.labelsize": 11,})

#Cores
COR_FERIADO     = "#4E79A7"
COR_NAO_FERIADO = "#F28E2B"

# --------- GERAL ---------

#Calculando media por feriado
feriados2 = (df[df["feriado?"]].groupby("nome_feriado")["consultas"].mean())

#Calculando media dos dias não feriados
mean_naofer_geral = df[~df["feriado?"]]["consultas"].mean()

#Plot
fig, ax = plt.subplots(figsize=(12,5))
ax.set_title("Média de Consultas por Feriado — Geral")
ax.set_xlabel("Feriados"); ax.set_ylabel("Número de Consultas")
ax.ticklabel_format(style="plain", axis="y")
idx = feriados2.index.tolist()
x = np.arange(len(idx)); w = 0.42
bars1 = ax.bar(x - w/2, feriados2.values, width=w, color=COR_FERIADO, edgecolor="black", linewidth=0.5, label="Feriado")
bars2 = ax.bar(x + w/2, np.full(len(idx), mean_naofer_geral), width=w, color=COR_NAO_FERIADO, edgecolor="black", linewidth=0.5, label="Não feriado")
ax.set_xticks(x); ax.set_xticklabels(idx, rotation=30, ha="right")
ax.legend(frameon=False, loc="upper left")
ax.axhline(mean_naofer_geral, ls="--", lw=1.4, color="#888")
plt.tight_layout(); plt.show()

# --------- POR SETOR ---------
for setor, df_setor in df.groupby("setor"):
    #Calculando média pro feriado
    feriados_setor = (df_setor[df_setor["feriado?"]].groupby("nome_feriado")["consultas"].mean())

    #Calculando media dos dias normais
    mean_naofer = df_setor[~df_setor["feriado?"]]["consultas"].mean()

    #Plot
    x = np.arange(len(feriados_setor)); w = 0.42
    fig, ax = plt.subplots(figsize=(12,5))
    ax.set_title(f"Média de Consultas por Feriado — Setor: {setor}")
    ax.set_xlabel("Feriados"); ax.set_ylabel("Número de Consultas")
    ax.ticklabel_format(style="plain", axis="y")
    bars1 = ax.bar(x - w/2, feriados_setor.values, width=w, color=COR_FERIADO, edgecolor="black", linewidth=0.5, label="Feriado")
    bars2 = ax.bar(x + w/2, np.full(len(feriados_setor), mean_naofer), width=w, color=COR_NAO_FERIADO, edgecolor="black", linewidth=0.5, label="Não feriado")
    ax.set_xticks(x); ax.set_xticklabels(feriados_setor.index, rotation=30, ha="right")
    ax.legend(frameon=False, loc="upper left")
    ax.axhline(mean_naofer, ls="--", lw=1.4, color="#888")
    plt.tight_layout(); plt.show()

# --------- POR HOSPITAL ---------
for hosp, df_hosp in df.groupby("cliente"):
    #Calculando a media por feriado
    feriados_hosp = (df_hosp[df_hosp["feriado?"]].groupby("nome_feriado")["consultas"].mean())

    #Calculando a media por dias normais
    mean_naofer = df_hosp[~df_hosp["feriado?"]]["consultas"].mean()

    #Plot
    x = np.arange(len(feriados_hosp)); w = 0.42
    fig, ax = plt.subplots(figsize=(12,5))
    ax.set_title(f"Média de Consultas por Feriado — Hospital: {hosp}")
    ax.set_xlabel("Feriados"); ax.set_ylabel("Número de Consultas")
    ax.ticklabel_format(style="plain", axis="y")
    bars1 = ax.bar(x - w/2, feriados_hosp.values, width=w, color=COR_FERIADO, edgecolor="black", linewidth=0.5, label="Feriado")
    bars2 = ax.bar(x + w/2, np.full(len(feriados_hosp), mean_naofer), width=w, color=COR_NAO_FERIADO, edgecolor="black", linewidth=0.5, label="Não feriado")
    ax.set_xticks(x); ax.set_xticklabels(feriados_hosp.index, rotation=30, ha="right")
    ax.legend(frameon=False, loc="upper left")
    ax.axhline(mean_naofer, ls="--", lw=1.4, color="#888")
    plt.tight_layout(); plt.show()

"""Por Ano"""

#Tema e tipografia
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120,"axes.titlesize": 18,"axes.labelsize": 13,"xtick.labelsize": 11,"ytick.labelsize": 11,})

#Cores
COR_GERAL = "#4E79A7"
COR_SETOR = "#F28E2B"
COR_HOSP  = "#59A14F"

# --------- GERAL ---------

ano = df_consultas.groupby(df_consultas["data"].dt.year)["consultas"].mean()

fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(ano.index.astype(str), ano.values, width=0.65, color=COR_GERAL, edgecolor="black", linewidth=0.5)
ax.set_title("Média de Consultas por Ano — Geral")
ax.set_xlabel("Ano"); ax.set_ylabel("Número de Consultas")
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.axhline(ano.mean(), ls="--", lw=1.4, color="#888", label="Média")

for b in bars:
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=10)

ax.legend(frameon=False, loc="upper left")
plt.tight_layout(); plt.show()

# --------- POR SETOR ---------

for setor, df_setor in df_consultas.groupby("setor"):
    ano_setor = df_setor.groupby(df_setor["data"].dt.year)["consultas"].mean()

    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(ano_setor.index.astype(str), ano_setor.values, width=0.65, color=COR_SETOR, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Ano — Setor: {setor}")
    ax.set_xlabel("Ano"); ax.set_ylabel("Número de Consultas")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.axhline(ano_setor.mean(), ls="--", lw=1.4, color="#888", label="Média")

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout(); plt.show()

# --------- POR HOSPITAL ---------

for unidade, df_un in df_consultas.groupby("unidade"):
    ano_un = df_un.groupby(df_un["data"].dt.year)["consultas"].mean()

    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.bar(ano_un.index.astype(str), ano_un.values, width=0.65, color=COR_HOSP, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Ano — Hospital: {unidade}")
    ax.set_xlabel("Ano"); ax.set_ylabel("Número de Consultas")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.axhline(ano_un.mean(), ls="--", lw=1.4, color="#888", label="Média")

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout(); plt.show()

"""##2.0 - Média de consultas por hora (por hospital e por setor)




"""

#Criando DF
mov = df_consultas.copy()

#Tema e tipografia
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi":120,"axes.titlesize":18,"axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11})

#Cores
COR_HOSP = "#59A14F"
COR_SET  = "#F28E2B"

# ------------------ POR HOSPITAL ------------------

#Calculando média
s_hosp = mov.groupby(["unidade","hora"], as_index=False)["consultas"].mean()

#Plot
for hosp, g in s_hosp.groupby("unidade"):
    serie = (g.set_index("hora")["consultas"].reindex(range(24)))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(serie.index, serie.values, width=0.65,color=COR_HOSP, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Hora — Hospital: {hosp}")
    ax.set_xlabel("Horário"); ax.set_ylabel("Consultas")
    ax.tick_params(axis="x", rotation=0); ax.ticklabel_format(style="plain", axis="y")
    ax.axhline(serie.mean(), ls="--", lw=1.4, color="#888", label="Média")

    #Picp
    h_peak = int(serie.idxmax()); y_peak = float(serie.max())
    ax.plot([h_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {h_peak}h")
    ax.text(h_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)

    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

# ------------------ POR SETOR (UNIDADE + SETOR) ------------------

#Calculando média

s_set = mov.groupby(["unidade","setor","hora"], as_index=False)["consultas"].mean()
s_set["unidade_setor"] = s_set["unidade"].astype(str).str.cat(s_set["setor"].astype(str), sep=" - ", na_rep="")

#Plot
for name, g in s_set.groupby("unidade_setor"):
    serie = (g.set_index("hora")["consultas"].reindex(range(24)))
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(serie.index, serie.values, width=0.65,color=COR_SET, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Média de Consultas por Hora — {name}")
    ax.set_xlabel("Horário"); ax.set_ylabel("Consultas")
    ax.tick_params(axis="x", rotation=0); ax.ticklabel_format(style="plain", axis="y")
    ax.axhline(serie.mean(), ls="--", lw=1.4, color="#888", label="Média")

    #Pico
    h_peak = int(serie.idxmax()); y_peak = float(serie.max())
    ax.plot([h_peak], [y_peak], marker="o", ms=8, color="#D62728", label=f"Pico: {h_peak}h")
    ax.text(h_peak, y_peak, f"{y_peak:.0f}", ha="center", va="bottom", fontsize=11)

    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

"""##3.0 - Surto de Dengue

"""

#Criando DF
df_dengue = df_consultas

#Checando Type das datas
df_dengue["data"] = pd.to_datetime(df_dengue["data"], dayfirst=True, errors="coerce")

#Periodo do surto
comeco_surto = pd.Timestamp("2024-02-01")
fim_surto    = pd.Timestamp("2024-04-30")
feriado = (df_dengue["data"] >= comeco_surto) & (df_dengue["data"] <= fim_surto)

feriado_sim = df_dengue.loc[feriado,  "consultas"].mean()
feriado_nao = df_dengue.loc[~feriado, "consultas"].mean()

#Plot
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120,"axes.titlesize": 18,"axes.labelsize": 13,"xtick.labelsize": 11,"ytick.labelsize": 11,})
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(["Durante o Surto (fev–abr/2024)", "Fora do Surto"],[feriado_sim, feriado_nao],width=0.6,color=["#6BAED6", "#F28E2B"],edgecolor="black",linewidth=0.6)
ax.set_ylabel("Média de consultas")
ax.set_title("Média de consultas — Durante o Surto vs. Fora do Surto")
ax.ticklabel_format(style="plain", axis="y")
ax.yaxis.grid(True, ls="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, max(feriado_sim, feriado_nao) * 1.20)

#Rótulos com o valor em cada barra
for b in bars:
    y = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, y, f"{y:.1f}",
            ha="center", va="bottom", fontsize=11)

#Δ% (Surto vs. Fora)
delta = 0 if feriado_nao == 0 else (feriado_sim - feriado_nao) / feriado_nao * 100
ax.text(0.5, max(feriado_sim, feriado_nao) * 1.12,f"Δ vs. Fora do Surto: {delta:+.1f}%",ha="center", va="bottom", fontsize=12, color="#333")
plt.tight_layout()
plt.show()



"""## 4.0 - Dias de Chuva

Dias de Chuva Geral
"""

#Garantir que data esta no formato certo
df_consultas["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")

#Filtrar para pegar apenas os anos desejados
df_2024 = df_consultas[df_consultas["data"].dt.year == 2024]

#Coletar informações sobre dias de chuva em São Paulo
LAT, LON = -23.55, -46.63
start, end = "2024-01-01", "2024-12-31"
url = ("https://archive-api.open-meteo.com/v1/era5"f"?latitude={LAT}&longitude={LON}"f"&start_date={start}&end_date={end}""&daily=precipitation_sum&timezone=auto")
wx = requests.get(url).json()
chuva_df = pd.DataFrame({"data": pd.to_datetime(wx["daily"]["time"]),"precip_mm": wx["daily"]["precipitation_sum"]})

#Marcar dias que choveu mais de 1mm
chuva_df["rained"] = chuva_df["precip_mm"] > 1

#Join nas tabelas
df_2024 = df_2024.merge(chuva_df[["data", "rained"]], on="data", how="left")

#Separar os dias que choveram dos que nao choveram
df_2024_chuva = df_2024[df_2024["rained"] == True]
df_2024_sem_chuva = df_2024[df_2024["rained"] == False]

#Escolher apenas os IDS que tem dias de chuva e dias sem chuva
id_finais = set(df_2024_chuva['idservico']).intersection(set(df_2024_sem_chuva['idservico']))

#Media Geral de chuva por hora
media_chuva_hora = df_2024_chuva.groupby("hora")["consultas"].mean()
media_sem_chuva_hora = df_2024_sem_chuva.groupby("hora")["consultas"].mean()

#Criar um DF com as medias dos dias de chuva e dos dias sem chuva
horas = range(24)
final_df = (pd.DataFrame({"Chuva":     media_chuva_hora,"Sem chuva": media_sem_chuva_hora}).reindex(horas))

#Plot
fig,ax = plt.subplots(figsize=(10,8))
fig.suptitle("Média de Consultas por Hora — Dias de Chuva vs. Sem Chuva")
final_df.plot(kind="bar", ax=ax, width=0.8,rot=0)
ax.set_xlabel("Hora do Dia")
ax.set_ylabel("Número de Consultas")
plt.tight_layout()

"""Dias de chuva de verão (Normalmente chuvas mais fortes acotecem no verão)"""

#Garantis que a coluna data esta no formato certo
df_consultas["data"] = pd.to_datetime(df_consultas["data"], dayfirst=True, errors="coerce")

#Usar a data do verão
start = pd.Timestamp("2024-12-21")
end   = pd.Timestamp("2025-03-21")

#Filtrar somente o período de verão
df_verao = df_consultas[df_consultas["data"].between(start, end, inclusive="both")].copy()

#Coletar dados de chuva em São Paulo no verão
LAT, LON = -23.55, -46.63
url = ("https://archive-api.open-meteo.com/v1/era5"f"?latitude={LAT}&longitude={LON}"f"&start_date={start.date()}&end_date={end.date()}""&daily=precipitation_sum&timezone=auto")
wx = requests.get(url).json()
chuva_df = pd.DataFrame({"data": pd.to_datetime(wx["daily"]["time"]),"precip_mm": wx["daily"]["precipitation_sum"]})

#Selecionar os dias que choveram
chuva_df["rained"] = chuva_df["precip_mm"] > 1

#Join nas DFs
df_verao = df_verao.merge(chuva_df[["data","rained"]], on="data", how="left")

#Separar dias de chuva e dias sem chuva
df_verao_chuva = df_verao[df_verao["rained"]]
df_verao_sem_chuva = df_verao[~df_verao["rained"]]

#Manter servicos que tem dias de chuva e dias sem chuva
ids_comuns = set(df_verao_chuva["idservico"]).intersection(df_verao_sem_chuva["idservico"])
df_verao_chuva = df_verao_chuva[df_verao_chuva["idservico"].isin(ids_comuns)]
df_verao_sem_chuva = df_verao_sem_chuva[df_verao_sem_chuva["idservico"].isin(ids_comuns)]

#Calcular as medias
media_chuva_hora = df_verao_chuva.groupby("hora")["consultas"].mean()
media_sem_chuva_hora = df_verao_sem_chuva.groupby("hora")["consultas"].mean()

#Criar a DF final para plotar
horas = range(24)
final_df = (pd.DataFrame({"Chuva":     media_chuva_hora,"Sem chuva": media_sem_chuva_hora}).reindex(horas))

#Plotar
fig, ax = plt.subplots(figsize=(10,8))
fig.suptitle("Média de Consultas por Hora — Verão 2024/25 (Chuva vs. Sem Chuva)")
final_df.plot(kind="bar", ax=ax, width=0.8, rot=0)
ax.set_xlabel("Hora do Dia")
ax.set_ylabel("Número de Consultas")
ax.ticklabel_format(style="plain", axis="y")
plt.tight_layout(); plt.show()

"""Dias de chuva no inverno"""

#Garantis que a coluna data esta no formato certo
df_consultas["data"] = pd.to_datetime(df_consultas["data"], dayfirst=True, errors="coerce")

#Usar a data do inverno
start = pd.Timestamp("2024-06-21")
end   = pd.Timestamp("2025-09-21")

#Filtrar somente o período de inverno
df_inverno = df_consultas[df_consultas["data"].between(start, end, inclusive="both")].copy()

#Coletar dados de chuva em São Paulo no inverno
LAT, LON = -23.55, -46.63
url = ("https://archive-api.open-meteo.com/v1/era5"f"?latitude={LAT}&longitude={LON}"f"&start_date={start.date()}&end_date={end.date()}""&daily=precipitation_sum&timezone=auto")
wx = requests.get(url).json()
chuva_df = pd.DataFrame({"data": pd.to_datetime(wx["daily"]["time"]),"precip_mm": wx["daily"]["precipitation_sum"]})

#Selecionar os dias que choveram
chuva_df["rained"] = chuva_df["precip_mm"] > 1

#Join nas DFs
df_inverno = df_inverno.merge(chuva_df[["data","rained"]], on="data", how="left")

#Separar dias de chuva e dias sem chuva
df_inverno_chuva = df_verao[df_verao["rained"]]
df_inverno_sem_chuva = df_verao[~df_verao["rained"]]

#Manter servicos que tem dias de chuva e dias sem chuva
ids_comuns = set(df_inverno_chuva["idservico"]).intersection(df_inverno_sem_chuva["idservico"])
df_inverno_chuva = df_inverno_chuva[df_inverno_chuva["idservico"].isin(ids_comuns)]
df_inverno_sem_chuva = df_inverno_sem_chuva[df_inverno_sem_chuva["idservico"].isin(ids_comuns)]

#Calcular as medias
media_chuva_hora = df_inverno_chuva.groupby("hora")["consultas"].mean()
media_sem_chuva_hora = df_inverno_sem_chuva.groupby("hora")["consultas"].mean()

#Criar a DF final para plotar
horas = range(24)
final_df = (pd.DataFrame({"Chuva":     media_chuva_hora,"Sem chuva": media_sem_chuva_hora}).reindex(horas))

#Plotar
fig, ax = plt.subplots(figsize=(10,8))
fig.suptitle("Média de Consultas por Hora — Verão 2024/25 (Chuva vs. Sem Chuva)")
final_df.plot(kind="bar", ax=ax, width=0.8, rot=0)
ax.set_xlabel("Hora do Dia")
ax.set_ylabel("Número de Consultas")
ax.ticklabel_format(style="plain", axis="y")
plt.tight_layout()

"""## 5.0 - Histogramas

"""

#Tema e tipografia
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120,"axes.titlesize": 18,"axes.labelsize": 13,"xtick.labelsize": 11,"ytick.labelsize": 11,})

COR_GERAL = "#4E79A7"
COR_SETOR = "#F28E2B"
COR_HOSP  = "#59A14F"
BORDA     = "black"
REF       = "#888"

# ---------- GERAL ----------

#Calculando a soma das consultas
daily_geral = df_consultas.groupby("data", as_index=False)["consultas"].sum()
vals = daily_geral["consultas"]  # sem drop/fill

#Plot
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(vals, bins="fd", edgecolor=BORDA, linewidth=0.6, color=COR_GERAL, alpha=0.9)

ax.set_title("GERAL — Distribuição de consultas diárias")
ax.set_xlabel("Consultas por dia"); ax.set_ylabel("Frequência (dias)")
ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

#Referências (média e mediana)
m  = float(np.mean(vals))
md = float(np.median(vals))
ax.axvline(m,  ls="--", lw=1.6, color=REF,   label=f"Média: {m:.0f}")
ax.axvline(md, ls=":",  lw=1.6, color="#555", label=f"Mediana: {md:.0f}")

ax.legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.show(); plt.close()


# ---------- POR SETOR ----------

#Calculando a soma das consultas por setor
daily_setor = df_consultas.groupby(["setor","data"], as_index=False)["consultas"].sum()

#Plot
for setor, g in daily_setor.groupby("setor"):
    vals = g["consultas"]  # sem drop/fill

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(vals, bins="fd", edgecolor=BORDA, linewidth=0.6, color=COR_SETOR, alpha=0.9)

    ax.set_title(f"{setor} — Distribuição de consultas diárias")
    ax.set_xlabel("Consultas por dia"); ax.set_ylabel("Frequência (dias)")
    ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    m  = float(np.mean(vals))
    md = float(np.median(vals))
    ax.axvline(m,  ls="--", lw=1.6, color=REF,   label=f"Média: {m:.0f}")
    ax.axvline(md, ls=":",  lw=1.6, color="#555", label=f"Mediana: {md:.0f}")

    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.show(); plt.close()


# ---------- POR HOSPITAL ----------

#Calculando a soma das consultas por unidade
daily_hosp = df_consultas.groupby(["unidade","data"], as_index=False)["consultas"].sum()

#Plot
for hosp, g in daily_hosp.groupby("unidade"):
    vals = g["consultas"]  # sem drop/fill

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(vals, bins="fd", edgecolor=BORDA, linewidth=0.6, color=COR_HOSP, alpha=0.9)

    ax.set_title(f"{hosp} — Distribuição de consultas diárias")
    ax.set_xlabel("Consultas por dia"); ax.set_ylabel("Frequência (dias)")
    ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    m  = float(np.mean(vals))
    md = float(np.median(vals))
    ax.axvline(m,  ls="--", lw=1.6, color=REF,   label=f"Média: {m:.0f}")
    ax.axvline(md, ls=":",  lw=1.6, color="#555", label=f"Mediana: {md:.0f}")

    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.show(); plt.close()

"""## 6.0 - Time Series

Time Series Surto de Dengue de 2024 - Alguns plots sao inuteis devido a falta de dados no periodo do surto
"""

#Estilo do plot
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 18, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11
})

#Surtos
SURTOS = [
    ("2022-03-01", "2022-05-31", "Surto 2022 (mar–mai)"),
    ("2023-03-15", "2023-05-31", "Surto 2023 (mar–mai)"),
    ("2024-02-01", "2024-04-30", "Surto 2024 (fev–abr)"),
]

#Cores
SURTO_COLORS = ["#6BAED6", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"]

#Criando a DF
df_ts = df_consultas.copy()

#Conferindo Type da data
df_ts["data"] = pd.to_datetime(df_ts["data"], dayfirst=True, errors="coerce")
df_ts = df_ts.dropna(subset=["data"]).sort_values("data")

# ---------- GERAL ----------
#Calculando o total por dia
daily = (df_ts.groupby("data", as_index=True)["consultas"].sum().rename("consultas_dia"))
daily = daily.asfreq("D")
mm7 = daily.rolling(7, min_periods=1).mean()

#Médias de referência dentro e fora dos os surtos combinados
mask_all = pd.Series(False, index=daily.index)
for (ini, fim, _rot) in SURTOS:
    ini_ts, fim_ts = pd.Timestamp(ini), pd.Timestamp(fim)
    mask_all |= (daily.index >= ini_ts) & (daily.index <= fim_ts)

media_surto = daily.loc[mask_all].mean()
media_fora  = daily.loc[~mask_all].mean()

#Plot
fig, ax = plt.subplots(figsize=(13,5))
ax.plot(daily.index, daily.values, lw=1.2, alpha=0.45, label="Diário (total)")
ax.plot(mm7.index,    mm7.values,    lw=2.2,              label="Média móvel (7d)")

for i, (ini, fim, rot) in enumerate(SURTOS):
    ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim),
               color=SURTO_COLORS[i % len(SURTO_COLORS)],
               alpha=0.18, label=rot)

if pd.notna(media_fora):  ax.axhline(media_fora,  color="#999",   ls="--", lw=1.2, alpha=0.9, label=f"Média fora ({media_fora:.1f})")
if pd.notna(media_surto): ax.axhline(media_surto, color="#1f77b4", ls="--", lw=1.2, alpha=0.9, label=f"Média nos surtos ({media_surto:.1f})")

ax.set_title("Consultas por dia (TOTAL) — GERAL — janelas de surtos de dengue")
ax.set_xlabel("Data"); ax.set_ylabel("Consultas por dia (total)")
ax.ticklabel_format(style="plain", axis="y")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", frameon=False, ncol=2)
plt.tight_layout(); plt.show()

# ---------- POR SETOR ----------

for setor, df_s in df_ts.groupby("setor"):
    #Calculanda a soma
    daily_s = (df_s.groupby("data", as_index=True)["consultas"].sum().rename("consultas_dia"))

    daily_s = daily_s.asfreq("D")
    mm7_s = daily_s.rolling(7, min_periods=1).mean()

    mask_all_s = pd.Series(False, index=daily_s.index)
    for (ini, fim, _rot) in SURTOS:
        ini_ts, fim_ts = pd.Timestamp(ini), pd.Timestamp(fim)
        mask_all_s |= (daily_s.index >= ini_ts) & (daily_s.index <= fim_ts)

    media_s_surto = daily_s.loc[mask_all_s].mean()
    media_s_fora  = daily_s.loc[~mask_all_s].mean()

    #Plot
    fig, ax = plt.subplots(figsize=(13,5))
    ax.plot(daily_s.index, daily_s.values, lw=1.2, alpha=0.45, label="Diário (total)")
    ax.plot(mm7_s.index,   mm7_s.values,   lw=2.2,              label="Média móvel (7d)")

    for i, (ini, fim, rot) in enumerate(SURTOS):
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim),
                   color=SURTO_COLORS[i % len(SURTO_COLORS)],
                   alpha=0.18, label=rot)

    if pd.notna(media_s_fora):  ax.axhline(media_s_fora,  color="#999",   ls="--", lw=1.2, alpha=0.9, label=f"Média fora ({media_s_fora:.1f})")
    if pd.notna(media_s_surto): ax.axhline(media_s_surto, color="#1f77b4", ls="--", lw=1.2, alpha=0.9, label=f"Média nos surtos ({media_s_surto:.1f})")

    ax.set_title(f"Consultas por dia (TOTAL) — SETOR: {setor}")
    ax.set_xlabel("Data"); ax.set_ylabel("Consultas por dia (total)")
    ax.ticklabel_format(style="plain", axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    plt.tight_layout(); plt.show()

# ---------- POR HOSPITAL ----------
for unidade, df_u in df_ts.groupby("unidade"):

    #Calculando a soma
    daily_u = (df_u.groupby("data", as_index=True)["consultas"].sum().rename("consultas_dia"))

    daily_u = daily_u.asfreq("D")
    mm7_u = daily_u.rolling(7, min_periods=1).mean()

    mask_all_u = pd.Series(False, index=daily_u.index)
    for (ini, fim, _rot) in SURTOS:
        ini_ts, fim_ts = pd.Timestamp(ini), pd.Timestamp(fim)
        mask_all_u |= (daily_u.index >= ini_ts) & (daily_u.index <= fim_ts)

    media_u_surto = daily_u.loc[mask_all_u].mean()
    media_u_fora  = daily_u.loc[~mask_all_u].mean()

    #Plot
    fig, ax = plt.subplots(figsize=(13,5))
    ax.plot(daily_u.index, daily_u.values, lw=1.2, alpha=0.45, label="Diário (total)")
    ax.plot(mm7_u.index,   mm7_u.values,   lw=2.2,              label="Média móvel (7d)")

    for i, (ini, fim, rot) in enumerate(SURTOS):
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim),
                   color=SURTO_COLORS[i % len(SURTO_COLORS)],
                   alpha=0.18, label=rot)

    if pd.notna(media_u_fora):  ax.axhline(media_u_fora,  color="#999",   ls="--", lw=1.2, alpha=0.9, label=f"Média fora ({media_u_fora:.1f})")
    if pd.notna(media_u_surto): ax.axhline(media_u_surto, color="#1f77b4", ls="--", lw=1.2, alpha=0.9, label=f"Média nos surtos ({media_u_surto:.1f})")

    ax.set_title(f"Consultas por dia (TOTAL) — HOSPITAL: {unidade}")
    ax.set_xlabel("Data"); ax.set_ylabel("Consultas por dia (total)")
    ax.ticklabel_format(style="plain", axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    plt.tight_layout(); plt.show()

"""## 7.0 - Heatmaps"""

#Nomeando dias da semana
df_consultas['dia_semana'] = df_consultas['data'].dt.day_name()

#Calculando media
df_heatmap = df_consultas.groupby(['dia_semana', 'hora'])['consultas'].mean().unstack("hora")

#Deixando os dias da semana em ordem
df_heatmap = df_heatmap.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# ---------- GERAL ----------

#Plot
plt.figure(figsize=(12,6))
sns.heatmap(df_heatmap,cbar_kws={"label": "Média de Consultas"},)
plt.title("Heatmap de Mêdia de Consultas por Dia da Semana e Hora")
plt.xlabel("Hora do Dia")
plt.ylabel("Dia da Semana")

# ---------- POR SETOR ----------

for setor,df_ps in df_consultas.groupby("setor"):
    #Calculando a média
    df_setor = df_ps.groupby(['dia_semana', 'hora'])['consultas'].mean().unstack("hora")
    df_setor = df_setor.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    #Plot
    plt.figure(figsize=(12,6))
    sns.heatmap(df_heatmap,cbar_kws={"label": "Média de Consultas"},)
    plt.title(f"Heatmap de Mêdia de Consultas por Dia da Semana e Hora - Setor: {setor}")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Dia da Semana")


# ---------- POR HOSPITAL ----------

#Plot
for hospital,df_hospital in df_consultas.groupby("cliente"):

    #Calculando a média
    df_hosp = df_hospital.groupby(['dia_semana', 'hora'])['consultas'].mean().unstack("hora")
    df_hosp = df_hosp.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    #Plot
    plt.figure(figsize=(12,6))
    sns.heatmap(df_heatmap,cbar_kws={"label": "Média de Consultas"},)
    plt.title(f"Heatmap de Mêdia de Consultas por Dia da Semana e Hora - Hospital: {hospital}")
    plt.xlabel("Hora do Dia")
    plt.ylabel("Dia da Semana")

"""## 8.0 - Dias da Semana x Finais de Semana"""

#Separa os dias da semana dos fins de semana
is_weekend = df_consultas['dia_semana'].isin(['Saturday','Sunday'])
df_consultas['tipo_dia'] = np.where(is_weekend, 'Fim de semana', 'Dia útil')

#Calculando a mdia
media_sem_vs_util = (df_consultas.groupby('tipo_dia', as_index=False)['consultas'].mean().rename(columns={'consultas':'media_consultas'}))

#Garante ordem
ordem = ["Dia útil", "Fim de semana"]
media_sem_vs_util["tipo_dia"] = pd.Categorical(media_sem_vs_util["tipo_dia"], categories=ordem, ordered=True)
media_sem_vs_util = media_sem_vs_util.sort_values("tipo_dia")

sns.set_theme(style="whitegrid", font_scale=1.1)

#Plot
plt.figure(figsize=(6,4), dpi=140)
ax = sns.barplot(data=media_sem_vs_util,x="tipo_dia",y="media_consultas",edgecolor="black",)
ax.set_title("Média diária de consultas — Dias úteis vs Fim de semana", pad=12)
ax.set_xlabel("")
ax.set_ylabel("Média de consultas por dia")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".")))

#Rótulos acima das barras
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f", padding=3)

#Espaço no topo para não cortar rótulos
ymax = media_sem_vs_util["media_consultas"].max()
ax.set_ylim(0, ymax * 1.15)

#Grid leve
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

"""# Model

## 1.0 - Backtest
"""

df = df_consultas.sort_values(by=["idservico", "data", "hora", "setor"]).reset_index(drop=True)

df["lag_24h"]  = df.groupby(["idservico", "setor"])["consultas"].shift(24*7 + 24)
df["lag_48h"]  = df.groupby(["idservico", "setor"])["consultas"].shift(24*7 + 48)
df["lag_72h"]  = df.groupby(["idservico", "setor"])["consultas"].shift(24*7 +72)

df["lag_1w"]   = df.groupby(["idservico", "setor"])["consultas"].shift(24 * 7)
df["lag_2w"]   = df.groupby(["idservico", "setor"])["consultas"].shift(24 * 7 * 2)
df["lag_3w"]   = df.groupby(["idservico", "setor"])["consultas"].shift(24 * 7 * 3)
df["lag_4w"]   = df.groupby(["idservico", "setor"])["consultas"].shift(24 * 7 * 4)

df["roll_mean_3h"] = (df.groupby(["idservico", "setor"])["consultas"].transform(lambda s: s.shift(24*7 +1).rolling(3).mean()))
df["roll_mean_6h"] = (df.groupby(["idservico", "setor"])["consultas"].transform(lambda s: s.shift(24*7 + 1).rolling(6).mean()))
df["roll_mean_24h"] = (df.groupby(["idservico", "setor"])["consultas"].transform(lambda s: s.shift(24*7 +1).rolling(24).mean()))

df["same_hour_mean_2w"] = df[["lag_1w", "lag_2w"]].mean(axis=1)
df["same_hour_mean_3w"] = df[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df["same_hour_mean_4w"] = df[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df["same_hour_wavg_4w"] = (0.4 * df["lag_1w"] +0.3 * df["lag_2w"] +0.2 * df["lag_3w"] +0.1 * df["lag_4w"])

df["idservico"] = df["idservico"].astype("category")
df["setor"] = df["setor"].astype("category")

df["idservico_cat"] = df["idservico"].cat.codes
df["setor_cat"] = df["setor"].cat.codes

df["Weekday"] = df["data"].dt.weekday
df["Month"] = df["data"].dt.month
df["day"] = df["data"].dt.day

df["weekofyear"] = df["data"].dt.isocalendar().week.astype(int)
df["dayofyear"] = df["data"].dt.dayofyear
df["quarter"] = df["data"].dt.quarter

#df["is_weekend"] = df["Weekday"].isin([5, 6]).astype(int) - piorou
#df["is_monday"] = (df["Weekday"] == 0).astype(int) - piorou
#df["is_friday"] = (df["Weekday"] == 4).astype(int) - piorou

#df["hour_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
#df["hour_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

df["weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

#df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
#df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

#df["is_night"] = df["hora"].between(0, 5).astype(int)
#df["is_morning"] = df["hora"].between(6, 11).astype(int)
#df["is_afternoon"] = df["hora"].between(12, 17).astype(int)
#df["is_evening"] = df["hora"].between(18, 23).astype(int)
#df["is_business_hour"] = df["hora"].between(8, 18).astype(int)
#df["is_lunch_time"] = df["hora"].between(12, 14).astype(int)

#df["lag_1h"] = g.shift(24*7 + 1)
#df["lag_2h"] = g.shift(24*7 + 2)
#df["lag_3h"] = g.shift(24*7 + 3)
#df["lag_6h"] = g.shift(24*7 + 6)
#df["lag_12h"] = g.shift(24*7 + 12)

#df["lag_96h"] = g.shift(24*7 + 96)
#df["lag_120h"] = g.shift(24*7 + 120)
#df["lag_144h"] = g.shift(24*7 + 144)

#df["lag_5w"] = g.shift(24 * 7 * 5)
#df["lag_6w"] = g.shift(24 * 7 * 6)
#df["lag_8w"] = g.shift(24 * 7 * 8)

#df["roll_mean_12h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(12).mean())
#df["roll_mean_48h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(48).mean())
#df["roll_mean_7d"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24 * 7).mean())

#df["roll_std_6h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(6).std())
#df["roll_std_24h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24).std())
#df["roll_std_7d"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24 * 7).std())

#df["roll_min_24h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24).min())
#df["roll_max_24h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24).max())
#df["roll_median_24h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24).median())

#df["roll_min_7d"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24 * 7).min())
#df["roll_max_7d"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24 * 7).max())
#df["roll_median_7d"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24 * 7).median())

#weekly_lags_2w = ["lag_1w", "lag_2w"]
#weekly_lags_3w = ["lag_1w", "lag_2w", "lag_3w"]
#weekly_lags_4w = ["lag_1w", "lag_2w", "lag_3w", "lag_4w"]

#df["same_hour_median_4w"] = df[weekly_lags_4w].median(axis=1)
#df["same_hour_std_4w"] = df[weekly_lags_4w].std(axis=1)
#df["same_hour_min_4w"] = df[weekly_lags_4w].min(axis=1)
#df["same_hour_max_4w"] = df[weekly_lags_4w].max(axis=1)
#df["same_hour_range_4w"] = df["same_hour_max_4w"] - df["same_hour_min_4w"]

#df["same_hour_wavg_4w_alt"] = (0.5 * df["lag_1w"] +0.25 * df["lag_2w"] +0.15 * df["lag_3w"] +0.10 * df["lag_4w"])

#df["diff_1h"] = df["lag_1h"] - df["lag_2h"]
#df["diff_3h"] = df["lag_1h"] - df["lag_3h"]
#df["diff_24h"] = df["lag_24h"] - df["lag_48h"]
#df["diff_1w"] = df["lag_1w"] - df["lag_2w"]
#df["diff_2w"] = df["lag_2w"] - df["lag_3w"]

#df["ratio_1h_2h"] = df["lag_1h"] / (df["lag_2h"] + 1)
#df["ratio_24h_48h"] = df["lag_24h"] / (df["lag_48h"] + 1)
#df["ratio_1w_2w"] = df["lag_1w"] / (df["lag_2w"] + 1)
#df["ratio_recent_week"] = df["roll_mean_24h"] / (df["same_hour_mean_4w"] + 1)

#df["ewm_6h"] = g.transform(lambda s: s.shift(24*7 + 1).ewm(span=6, adjust=False).mean())
#df["ewm_24h"] = g.transform(lambda s: s.shift(24*7 + 1).ewm(span=24, adjust=False).mean())
#df["ewm_7d"] = g.transform(lambda s: s.shift(24*7 + 1).ewm(span=24*7, adjust=False).mean())

#df["was_zero_1h"] = (df["lag_1h"] == 0).astype(int)
#df["was_zero_24h"] = (df["lag_24h"] == 0).astype(int)
#df["was_zero_1w"] = (df["lag_1w"] == 0).astype(int)

#df["zero_count_24h"] = g.transform(lambda s: s.shift(24*7 + 1).rolling(24).apply(lambda x: (x == 0).sum(), raw=True))
#df["zero_rate_24h"] = df["zero_count_24h"] / 24

#df["is_recent_peak_24h"] = (df["lag_1h"] >= df["roll_max_24h"]).astype(int)
#df["distance_from_24h_max"] = df["roll_max_24h"] - df["lag_1h"]


features = ["idservico_cat", "setor_cat", "hora", "Weekday", "Month", "day","lag_1w","lag_2w","lag_3w","lag_4w","lag_24h","lag_48h","lag_72h",
            "roll_mean_3h","roll_mean_6h","roll_mean_24h","same_hour_mean_2w","same_hour_mean_3w","same_hour_mean_4w","same_hour_wavg_4w",
            "weekofyear","dayofyear","quarter","weekday_sin","weekday_cos"]

target = "consultas"

df_model = df.dropna(subset=features + [target]).copy()

df_train = df_model[(df_model["data"] >= "2021-01-01") & (df_model["data"] < "2024-01-01")].copy()
df_val = df_model[(df_model["data"] >= "2024-01-01") & (df_model["data"] < "2025-01-01")].copy()
df_test = df_model[(df_model["data"] >= "2025-01-01") & (df_model["data"] < "2026-01-01")].copy()

X_train = df_train[features]
y_train = df_train[target]

X_val = df_val[features]
y_val = df_val[target]

X_test = df_test[features]
y_test = df_test[target]

xgb_grid = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
    "reg_lambda": [0.1, 1, 2, 3, 5, 10],
    "gamma": [0, 0.05, 0.1, 0.3],
}

best_score = float("inf")
best_params = None
best_n_estimators = None

for i, params in enumerate(ParameterSampler(xgb_grid, n_iter=10, random_state=42), start=1):
    print(f"Training model {i}/40")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae",
        early_stopping_rounds=100,
        **params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred = model.predict(X_val)
    val_pred = np.clip(val_pred, 0, None)

    val_mae = mean_absolute_error(y_val, val_pred)

    print(f"Validation MAE: {val_mae:.4f}")

    if val_mae < best_score:
        best_score = val_mae
        best_params = params
        best_n_estimators = model.best_iteration + 1

print("\nBest validation MAE:", best_score)
print("Best parameters:")
print(best_params)
print("Best number of trees:", best_n_estimators)

df_train_final = df_model[(df_model["data"] >= "2021-01-01") & (df_model["data"] < "2025-01-01")].copy()

X_train_final = df_train_final[features]
y_train_final = df_train_final[target]

final_model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=best_n_estimators,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="mae",
    **best_params
)

final_model.fit(X_train_final, y_train_final)

print("Final model trained.")

X_test = df_test[features]
y_test = df_test[target]

y_pred = final_model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Final 2025 MAE:  {mae:.4f}")
print(f"Final 2025 MSE:  {mse:.4f}")
print(f"Final 2025 RMSE: {rmse:.4f}")

"""## 2.0 - Benchmark

Benchmark 1 - XGBoost
"""

df_eval = df_consultas.copy()
df_eval["data"] = pd.to_datetime(df_eval["data"])

df_eval = df_eval[["idservico", "data", "hora", "consultas","unidade", "setor", "servico", "cliente"]].copy()

def add_week(df, col_name, delta_days):
    ref = df[["idservico", "setor", "hora", "data", "consultas"]].copy()
    ref["data"] = ref["data"] + pd.Timedelta(days=delta_days)
    ref = ref.rename(columns={"consultas": col_name})

    return df.merge(ref[["idservico", 'setor',"hora", "data", col_name]],on=["idservico", 'setor',"hora", "data"],how="left")

df_eval = add_week(df_eval, "w-1", 7)
df_eval = add_week(df_eval, "w-2", 14)

for i in range(1, 6):
    df_eval = add_week(df_eval, f"w+{i}", -7 * i)

df_eval = df_eval.dropna(subset=["w-1", "w-2", "w+1", "w+2", "w+3", "w+4", "w+5"]).reset_index(drop=True)

df_eval["p_w+1_bench"] = df_eval[["consultas", "w-1", "w-2"]].mean(axis=1)
df_eval["p_w+2_bench"] = df_eval[["consultas", "w-1", "p_w+1_bench"]].mean(axis=1)
df_eval["p_w+3_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench"]].mean(axis=1)
df_eval["p_w+4_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench"]].mean(axis=1)
df_eval["p_w+5_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench"]].mean(axis=1)

df_eval = df_eval.sort_values(["idservico", "setor", "data", "hora"]).reset_index(drop=True)
grp = df_eval.groupby(["idservico", "setor"])["consultas"]

df_eval["lag_24h"] = grp.shift(24*7 + 24)
df_eval["lag_48h"] = grp.shift(24*7 + 48)
df_eval["lag_72h"] = grp.shift(24*7 + 72)

df_eval["lag_1w"] = grp.shift(24 * 7)
df_eval["lag_2w"] = grp.shift(24 * 7 * 2)
df_eval["lag_3w"] = grp.shift(24 * 7 * 3)
df_eval["lag_4w"] = grp.shift(24 * 7 * 4)

df_eval["roll_mean_3h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(3).mean())
df_eval["roll_mean_6h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(6).mean())
df_eval["roll_mean_24h"] = grp.transform(lambda s: s.shift(24*7 + 1).rolling(24).mean())

df_eval["same_hour_mean_2w"] = df_eval[["lag_1w", "lag_2w"]].mean(axis=1)
df_eval["same_hour_mean_3w"] = df_eval[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df_eval["same_hour_mean_4w"] = df_eval[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df_eval["same_hour_wavg_4w"] = (0.4 * df_eval["lag_1w"] +0.3 * df_eval["lag_2w"] +0.2 * df_eval["lag_3w"] +0.1 * df_eval["lag_4w"])

df_eval["idservico"] = df_eval["idservico"].astype("category")
df_eval["setor"] = df_eval["setor"].astype("category")

df_eval["idservico_cat"] = df_eval["idservico"].cat.codes
df_eval["setor_cat"] = df_eval["setor"].cat.codes

df_eval["Weekday"] = df_eval["data"].dt.weekday
df_eval["Month"] = df_eval["data"].dt.month
df_eval["day"] = df_eval["data"].dt.day

features = ["idservico_cat", "setor_cat", "hora", "Weekday", "Month", "day","lag_1w", "lag_2w", "lag_3w", "lag_4w","lag_24h", "lag_48h", "lag_72h",
    "roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w", "same_hour_mean_3w", "same_hour_mean_4w", "same_hour_wavg_4w"]

needed_cols = features + ["w+1", "w+2", "w+3", "w+4", "w+5","p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench", "p_w+5_bench"]

df_eval = df_eval.dropna(subset=needed_cols).reset_index(drop=True)

df_train = df_eval[(df_eval["data"] >= "2021-01-01") & (df_eval["data"] < "2025-01-01")].copy()
df_test  = df_eval[(df_eval["data"] >= "2025-01-01") & (df_eval["data"] < "2026-01-01")].copy()

targets = {"w+1": "w+1","w+2": "w+2","w+3": "w+3","w+4": "w+4","w+5": "w+5",}

models = {}

for horizon, target_col in targets.items():
    X_train = df_train[features]
    y_train = df_train[target_col]

    X_test = df_test[features]
    y_test = df_test[target_col]

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    pred_col = f"p_{horizon}_xgb"
    df_test[pred_col] = model.predict(X_test)
    df_test[pred_col] = np.clip(df_test[pred_col], 0, None)

    models[horizon] = model
    print(f"{horizon} done")

horizontes = ["w+1", "w+2", "w+3", "w+4", "w+5"]

resultados = []

for h in horizontes:
    y_true = df_test[h]

    y_pred_bench = df_test[f"p_{h}_bench"]
    mae_bench = mean_absolute_error(y_true, y_pred_bench)
    mse_bench = mean_squared_error(y_true, y_pred_bench)
    rmse_bench = np.sqrt(mse_bench)

    y_pred_xgb = df_test[f"p_{h}_xgb"]
    mae_xgb = mean_absolute_error(y_true, y_pred_xgb)
    mse_xgb = mean_squared_error(y_true, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)

    resultados.append({
        "horizonte": h,
        "MAE_benchmark": mae_bench,
        "MSE_benchmark": mse_bench,
        "RMSE_benchmark": rmse_bench,
        "MAE_model": mae_xgb,
        "MSE_model": mse_xgb,
        "RMSE_model": rmse_xgb,
        "diff_MAE": mae_xgb - mae_bench,
        "diff_MSE": mse_xgb - mse_bench,
        "diff_RMSE": rmse_xgb - rmse_bench,
    })

resultado_metricas = pd.DataFrame(resultados)
print(resultado_metricas)

"""Benchmark 2 - Linear Regression (Sem Intercetp)


"""

df2 = df_consultas.copy()
df2["data"] = pd.to_datetime(df2["data"])

df2 = df2.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

keys = ["idservico", "setor", "data", "hora"]

base = df2[keys + ["consultas"]].copy()

def add_week(df2, base, col_name, days):
    ref = base.copy()
    ref["data"] = ref["data"] + timedelta(days=days)
    ref = ref.rename(columns={"consultas": col_name})
    return df2.merge(
        ref[keys + [col_name]],
        on=keys,
        how="left"
    )

df2 = add_week(df2, base, "w-4", 28)
df2 = add_week(df2, base, "w-3", 21)
df2 = add_week(df2, base, "w-2", 14)
df2 = add_week(df2, base, "w-1", 7)

df2 = add_week(df2, base, "w+1", -7)
df2 = add_week(df2, base, "w+2", -14)
df2 = add_week(df2, base, "w+3", -21)
df2 = add_week(df2, base, "w+4", -28)
df2 = add_week(df2, base, "w+5", -35)

df2 = df2.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

grp = df2.groupby(["idservico", "setor"])["consultas"]

df2["lag_24h"] = grp.shift(24 * 7 + 24)
df2["lag_48h"] = grp.shift(24 * 7 + 48)
df2["lag_72h"] = grp.shift(24 * 7 + 72)

df2["lag_1w"] = grp.shift(24 * 7)
df2["lag_2w"] = grp.shift(24 * 7 * 2)
df2["lag_3w"] = grp.shift(24 * 7 * 3)
df2["lag_4w"] = grp.shift(24 * 7 * 4)

df2["roll_mean_3h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(3).mean())
df2["roll_mean_6h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(6).mean())
df2["roll_mean_24h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(24).mean())

df2["same_hour_mean_2w"] = df2[["lag_1w", "lag_2w"]].mean(axis=1)
df2["same_hour_mean_3w"] = df2[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df2["same_hour_mean_4w"] = df2[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df2["same_hour_wavg_4w"] = (0.4 * df2["lag_1w"] + 0.3 * df2["lag_2w"] + 0.2 * df2["lag_3w"] + 0.1 * df2["lag_4w"])

feature_cols = ["consultas","w-1", "w-2", "w-3", "w-4","lag_24h", "lag_48h", "lag_72h","lag_1w",
                "lag_2w", "lag_3w", "lag_4w","roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w",
                "same_hour_mean_3w", "same_hour_mean_4w","same_hour_wavg_4w"]

target_cols = ["w+1", "w+2", "w+3", "w+4", "w+5"]

needed_cols = feature_cols + target_cols

df2 = df2.dropna(subset=needed_cols).reset_index(drop=True)

df_train = df2[(df2["data"] >= "2021-01-01") & (df2["data"] < "2025-01-01")].copy()
df_test = df2[(df2["data"] >= "2025-01-01") &(df2["data"] < "2026-01-01")].copy()

modelos = {}

for h in range(1, 6):
    target_col = f"w+{h}"
    pred_col = f"p_w+{h}"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]

    modelo = LinearRegression(fit_intercept=False)
    modelo.fit(X_train, y_train)

    df_test[pred_col] = modelo.predict(X_test)
    df_test[pred_col] = df_test[pred_col].clip(lower=0)

    modelos[target_col] = modelo

    mae = mean_absolute_error(df_test[target_col], df_test[pred_col])
    rmse = np.sqrt(mean_squared_error(df_test[target_col], df_test[pred_col]))

    print(f"\nHorizonte {target_col}")
    print(f"Intercepto: {modelo.intercept_:.4f}")
    print(f"MAE teste: {mae:.4f}")
    print(f"RMSE teste: {rmse:.4f}")

    pesos = pd.DataFrame({
        "feature": feature_cols,
        "peso": modelo.coef_
    }).sort_values(by="peso", ascending=False)

    print("\nPesos do modelo:")
    print(pesos)

benchmark_col = "same_hour_mean_3w"

resultados_erros = []

for h in range(1, 6):
    real_col = f"w+{h}"
    pred_col = f"p_w+{h}"
    horizonte = f"w+{h}"

    y_real = df_test[real_col]
    y_benchmark = df_test[benchmark_col]
    y_model = df_test[pred_col]

    mae_benchmark = mean_absolute_error(y_real, y_benchmark)
    mse_benchmark = mean_squared_error(y_real, y_benchmark)
    rmse_benchmark = np.sqrt(mse_benchmark)

    mae_model = mean_absolute_error(y_real, y_model)
    mse_model = mean_squared_error(y_real, y_model)
    rmse_model = np.sqrt(mse_model)

    resultados_erros.append({
        "horizonte": horizonte,
        "MAE_benchmark": mae_benchmark,
        "MSE_benchmark": mse_benchmark,
        "RMSE_benchmark": rmse_benchmark,
        "MAE_model": mae_model,
        "MSE_model": mse_model,
        "RMSE_model": rmse_model,
        "diff MAE": mae_model - mae_benchmark,
        "diff MSE": mse_model - mse_benchmark,
        "diff RMSE": rmse_model - rmse_benchmark
    })

df_erros = pd.DataFrame(resultados_erros)

print("\nResumo dos erros no teste:")
print(df_erros)

cols_to_show = keys + [
    "consultas",
    "w-4", "w-3", "w-2", "w-1",
    "same_hour_mean_3w",
    "w+1", "p_w+1",
    "w+2", "p_w+2",
    "w+3", "p_w+3",
    "w+4", "p_w+4",
    "w+5", "p_w+5"
]

print("\nExemplo do dataframe final de teste:")
print(df_test[cols_to_show].head())

df2_resultado = df_test.copy()

"""Benchmark 2 - Linear Regression (Com Intercept)"""

df2 = df_consultas.copy()
df2["data"] = pd.to_datetime(df2["data"])

df2 = df2.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

keys = ["idservico", "setor", "data", "hora"]

base = df2[keys + ["consultas"]].copy()

def add_week(df2, base, col_name, days):
    ref = base.copy()
    ref["data"] = ref["data"] + timedelta(days=days)
    ref = ref.rename(columns={"consultas": col_name})
    return df2.merge(
        ref[keys + [col_name]],
        on=keys,
        how="left"
    )

df2 = add_week(df2, base, "w-4", 28)
df2 = add_week(df2, base, "w-3", 21)
df2 = add_week(df2, base, "w-2", 14)
df2 = add_week(df2, base, "w-1", 7)

df2 = add_week(df2, base, "w+1", -7)
df2 = add_week(df2, base, "w+2", -14)
df2 = add_week(df2, base, "w+3", -21)
df2 = add_week(df2, base, "w+4", -28)
df2 = add_week(df2, base, "w+5", -35)

df2 = df2.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

grp = df2.groupby(["idservico", "setor"])["consultas"]

df2["lag_24h"] = grp.shift(24 * 7 + 24)
df2["lag_48h"] = grp.shift(24 * 7 + 48)
df2["lag_72h"] = grp.shift(24 * 7 + 72)

df2["lag_1w"] = grp.shift(24 * 7)
df2["lag_2w"] = grp.shift(24 * 7 * 2)
df2["lag_3w"] = grp.shift(24 * 7 * 3)
df2["lag_4w"] = grp.shift(24 * 7 * 4)

df2["roll_mean_3h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(3).mean())
df2["roll_mean_6h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(6).mean())
df2["roll_mean_24h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(24).mean())

df2["same_hour_mean_2w"] = df2[["lag_1w", "lag_2w"]].mean(axis=1)
df2["same_hour_mean_3w"] = df2[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df2["same_hour_mean_4w"] = df2[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df2["same_hour_wavg_4w"] = (0.4 * df2["lag_1w"] + 0.3 * df2["lag_2w"] + 0.2 * df2["lag_3w"] + 0.1 * df2["lag_4w"])

feature_cols = ["consultas","w-1", "w-2", "w-3", "w-4","lag_24h", "lag_48h", "lag_72h","lag_1w",
                "lag_2w", "lag_3w", "lag_4w","roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w",
                "same_hour_mean_3w", "same_hour_mean_4w","same_hour_wavg_4w"]

target_cols = ["w+1", "w+2", "w+3", "w+4", "w+5"]

needed_cols = feature_cols + target_cols

df2 = df2.dropna(subset=needed_cols).reset_index(drop=True)

df_train = df2[(df2["data"] >= "2021-01-01") &(df2["data"] < "2025-01-01")].copy()

df_test = df2[(df2["data"] >= "2025-01-01") &(df2["data"] < "2026-01-01")].copy()

modelos = {}

for h in range(1, 6):
    target_col = f"w+{h}"
    pred_col = f"p_w+{h}"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]

    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(X_train, y_train)

    df_test[pred_col] = modelo.predict(X_test)
    df_test[pred_col] = df_test[pred_col].clip(lower=0)

    modelos[target_col] = modelo

    mae = mean_absolute_error(df_test[target_col], df_test[pred_col])
    rmse = np.sqrt(mean_squared_error(df_test[target_col], df_test[pred_col]))

    print(f"\nHorizonte {target_col}")
    print(f"Intercepto: {modelo.intercept_:.4f}")
    print(f"MAE teste: {mae:.4f}")
    print(f"RMSE teste: {rmse:.4f}")

    pesos = pd.DataFrame({
        "feature": feature_cols,
        "peso": modelo.coef_
    }).sort_values(by="peso", ascending=False)

    print("\nPesos do modelo:")
    print(pesos)

benchmark_col = "same_hour_mean_3w"

resultados_erros = []

for h in range(1, 6):
    real_col = f"w+{h}"
    pred_col = f"p_w+{h}"
    horizonte = f"w+{h}"

    y_real = df_test[real_col]
    y_benchmark = df_test[benchmark_col]
    y_model = df_test[pred_col]

    mae_benchmark = mean_absolute_error(y_real, y_benchmark)
    mse_benchmark = mean_squared_error(y_real, y_benchmark)
    rmse_benchmark = np.sqrt(mse_benchmark)

    mae_model = mean_absolute_error(y_real, y_model)
    mse_model = mean_squared_error(y_real, y_model)
    rmse_model = np.sqrt(mse_model)

    resultados_erros.append({
        "horizonte": horizonte,
        "MAE_benchmark": mae_benchmark,
        "MSE_benchmark": mse_benchmark,
        "RMSE_benchmark": rmse_benchmark,
        "MAE_model": mae_model,
        "MSE_model": mse_model,
        "RMSE_model": rmse_model,
        "diff MAE": mae_model - mae_benchmark,
        "diff MSE": mse_model - mse_benchmark,
        "diff RMSE": rmse_model - rmse_benchmark
    })

df_erros = pd.DataFrame(resultados_erros)

print("\nResumo dos erros no teste:")
print(df_erros)

cols_to_show = keys + [
    "consultas",
    "w-4", "w-3", "w-2", "w-1",
    "same_hour_mean_3w",
    "w+1", "p_w+1",
    "w+2", "p_w+2",
    "w+3", "p_w+3",
    "w+4", "p_w+4",
    "w+5", "p_w+5"
]

print("\nExemplo do dataframe final de teste:")
print(df_test[cols_to_show].head())

df2_resultado_intercept = df_test.copy()

"""Benchmark 3 - Linear Regression (Sem Intercept)

"""

df3 = df_consultas.copy()

df3["data"] = pd.to_datetime(df3["data"])

df3 = df3.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

keys = ["idservico", "setor", "data", "hora"]

base = df3[keys + ["consultas"]].copy()

def add_week(df3, base, col_name, days):
    ref = base.copy()
    ref["data"] = ref["data"] + timedelta(days=days)
    ref = ref.rename(columns={"consultas": col_name})
    return df3.merge(
        ref[keys + [col_name]],
        on=keys,
        how="left"
    )

df3 = add_week(df3, base, "w-4", 28)
df3 = add_week(df3, base, "w-3", 21)
df3 = add_week(df3, base, "w-2", 14)
df3 = add_week(df3, base, "w-1", 7)

df3 = add_week(df3, base, "w+1", -7)
df3 = add_week(df3, base, "w+2", -14)
df3 = add_week(df3, base, "w+3", -21)
df3 = add_week(df3, base, "w+4", -28)
df3 = add_week(df3, base, "w+5", -35)

df3 = df3.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

grp = df3.groupby(["idservico", "setor"])["consultas"]

df3["lag_24h"] = grp.shift(24 * 7 + 24)
df3["lag_48h"] = grp.shift(24 * 7 + 48)
df3["lag_72h"] = grp.shift(24 * 7 + 72)

df3["lag_1w"] = grp.shift(24 * 7)
df3["lag_2w"] = grp.shift(24 * 7 * 2)
df3["lag_3w"] = grp.shift(24 * 7 * 3)
df3["lag_4w"] = grp.shift(24 * 7 * 4)

df3["roll_mean_3h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(3).mean())
df3["roll_mean_6h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(6).mean())
df3["roll_mean_24h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(24).mean())

df3["same_hour_mean_2w"] = df3[["lag_1w", "lag_2w"]].mean(axis=1)
df3["same_hour_mean_3w"] = df3[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df3["same_hour_mean_4w"] = df3[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df3["same_hour_wavg_4w"] = (0.4 * df3["lag_1w"] + 0.3 * df3["lag_2w"] + 0.2 * df3["lag_3w"] + 0.1 * df3["lag_4w"])

feature_cols = ["consultas","w-1", "w-2", "w-3", "w-4","lag_24h", "lag_48h", "lag_72h","lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w", "same_hour_mean_3w", "same_hour_mean_4w","same_hour_wavg_4w"]

target_cols = ["w+1", "w+2", "w+3", "w+4", "w+5"]

needed_cols = feature_cols + target_cols

df3 = df3.dropna(subset=needed_cols).reset_index(drop=True)

df3_train = df3[(df3["data"] >= "2021-01-01") & (df3["data"] < "2025-01-01")].copy()
df3_test = df3[(df3["data"] >= "2025-01-01") & (df3["data"] < "2026-01-01")].copy()

modelos_df3 = {}
pesos_df3 = []

for (servico, setor), df_grupo_train in df3_train.groupby(["idservico", "setor"]):
    df_grupo_test = df3_test[(df3_test["idservico"] == servico) & (df3_test["setor"] == setor)].copy()

    if len(df_grupo_test) == 0:
        continue

    df_grupo_train = df_grupo_train.copy()

    modelos_df3[(servico, setor)] = {}

    for h in range(1, 6):
        target_col = f"w+{h}"
        pred_col = f"p_w+{h}"

        X_train = df_grupo_train[feature_cols]
        y_train = df_grupo_train[target_col]
        X_test = df_grupo_test[feature_cols]

        modelo = LinearRegression(fit_intercept=False)
        modelo.fit(X_train, y_train)

        df_grupo_test[pred_col] = modelo.predict(X_test)
        df_grupo_test[pred_col] = df_grupo_test[pred_col].clip(lower=0)

        modelos_df3[(servico, setor)][target_col] = modelo

        for feature, peso in zip(feature_cols, modelo.coef_):
            pesos_df3.append({
                "idservico": servico,
                "setor": setor,
                "horizonte": target_col,
                "feature": feature,
                "peso": peso
            })

    df3_test.loc[df_grupo_test.index, ["p_w+1", "p_w+2", "p_w+3", "p_w+4", "p_w+5"]] = df_grupo_test[
        ["p_w+1", "p_w+2", "p_w+3", "p_w+4", "p_w+5"]
    ]

df_pesos_df3 = pd.DataFrame(pesos_df3)

benchmark_col = "same_hour_mean_3w"

resultados_erros_df3 = []

for h in range(1, 6):
    real_col = f"w+{h}"
    pred_col = f"p_w+{h}"
    horizonte = f"w+{h}"

    df_eval = df3_test.dropna(subset=[real_col, pred_col, benchmark_col]).copy()

    y_real = df_eval[real_col]
    y_benchmark = df_eval[benchmark_col]
    y_model = df_eval[pred_col]

    mae_benchmark = mean_absolute_error(y_real, y_benchmark)
    mse_benchmark = mean_squared_error(y_real, y_benchmark)
    rmse_benchmark = np.sqrt(mse_benchmark)

    mae_model = mean_absolute_error(y_real, y_model)
    mse_model = mean_squared_error(y_real, y_model)
    rmse_model = np.sqrt(mse_model)

    resultados_erros_df3.append({
        "horizonte": horizonte,
        "MAE_benchmark": mae_benchmark,
        "MSE_benchmark": mse_benchmark,
        "RMSE_benchmark": rmse_benchmark,
        "MAE_model": mae_model,
        "MSE_model": mse_model,
        "RMSE_model": rmse_model,
        "diff MAE": mae_model - mae_benchmark,
        "diff MSE": mse_model - mse_benchmark,
        "diff RMSE": rmse_model - rmse_benchmark
    })

df_erros_df3 = pd.DataFrame(resultados_erros_df3)

print("\nResumo dos erros do df3:")
print(df_erros_df3)

print("\nPesos do modelo por idservico, setor e horizonte:")
print(df_pesos_df3)

cols_to_show = keys + [
    "consultas",
    "w-4", "w-3", "w-2", "w-1",
    "same_hour_mean_3w",
    "w+1", "p_w+1",
    "w+2", "p_w+2",
    "w+3", "p_w+3",
    "w+4", "p_w+4",
    "w+5", "p_w+5"
]

print("\nExemplo do dataframe final df3:")
print(df3_test[cols_to_show].head())

df3_resultado = df3_test.copy()

"""Benchmark 3 - Linear Regression (Com Intercept)"""

df3 = df_consultas.copy()

df3["data"] = pd.to_datetime(df3["data"])

df3 = df3.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

keys = ["idservico", "setor", "data", "hora"]

base = df3[keys + ["consultas"]].copy()

def add_week(df3, base, col_name, days):
    ref = base.copy()
    ref["data"] = ref["data"] + timedelta(days=days)
    ref = ref.rename(columns={"consultas": col_name})
    return df3.merge(
        ref[keys + [col_name]],
        on=keys,
        how="left"
    )

df3 = add_week(df3, base, "w-4", 28)
df3 = add_week(df3, base, "w-3", 21)
df3 = add_week(df3, base, "w-2", 14)
df3 = add_week(df3, base, "w-1", 7)

df3 = add_week(df3, base, "w+1", -7)
df3 = add_week(df3, base, "w+2", -14)
df3 = add_week(df3, base, "w+3", -21)
df3 = add_week(df3, base, "w+4", -28)
df3 = add_week(df3, base, "w+5", -35)

df3 = df3.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

grp = df3.groupby(["idservico", "setor"])["consultas"]

df3["lag_24h"] = grp.shift(24 * 7 + 24)
df3["lag_48h"] = grp.shift(24 * 7 + 48)
df3["lag_72h"] = grp.shift(24 * 7 + 72)

df3["lag_1w"] = grp.shift(24 * 7)
df3["lag_2w"] = grp.shift(24 * 7 * 2)
df3["lag_3w"] = grp.shift(24 * 7 * 3)
df3["lag_4w"] = grp.shift(24 * 7 * 4)

df3["roll_mean_3h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(3).mean())
df3["roll_mean_6h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(6).mean())
df3["roll_mean_24h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(24).mean())

df3["same_hour_mean_2w"] = df3[["lag_1w", "lag_2w"]].mean(axis=1)
df3["same_hour_mean_3w"] = df3[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df3["same_hour_mean_4w"] = df3[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df3["same_hour_wavg_4w"] = (0.4 * df3["lag_1w"] + 0.3 * df3["lag_2w"] + 0.2 * df3["lag_3w"] + 0.1 * df3["lag_4w"])

feature_cols = ["consultas","w-1", "w-2", "w-3", "w-4","lag_24h", "lag_48h", "lag_72h","lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w", "same_hour_mean_3w", "same_hour_mean_4w","same_hour_wavg_4w"]

target_cols = ["w+1", "w+2", "w+3", "w+4", "w+5"]

needed_cols = feature_cols + target_cols

df3 = df3.dropna(subset=needed_cols).reset_index(drop=True)

df3_train = df3[(df3["data"] >= "2021-01-01") & (df3["data"] < "2025-01-01")].copy()
df3_test = df3[(df3["data"] >= "2025-01-01") & (df3["data"] < "2026-01-01")].copy()

modelos_df3 = {}
pesos_df3 = []

for (servico, setor), df_grupo_train in df3_train.groupby(["idservico", "setor"]):
    df_grupo_test = df3_test[(df3_test["idservico"] == servico) & (df3_test["setor"] == setor)].copy()

    if len(df_grupo_test) == 0:
        continue

    df_grupo_train = df_grupo_train.copy()

    modelos_df3[(servico, setor)] = {}

    for h in range(1, 6):
        target_col = f"w+{h}"
        pred_col = f"p_w+{h}"

        X_train = df_grupo_train[feature_cols]
        y_train = df_grupo_train[target_col]
        X_test = df_grupo_test[feature_cols]

        modelo = LinearRegression(fit_intercept=True)
        modelo.fit(X_train, y_train)

        df_grupo_test[pred_col] = modelo.predict(X_test)
        df_grupo_test[pred_col] = df_grupo_test[pred_col].clip(lower=0)

        modelos_df3[(servico, setor)][target_col] = modelo

        for feature, peso in zip(feature_cols, modelo.coef_):
            pesos_df3.append({
                "idservico": servico,
                "setor": setor,
                "horizonte": target_col,
                "feature": feature,
                "peso": peso
            })

    df3_test.loc[df_grupo_test.index, ["p_w+1", "p_w+2", "p_w+3", "p_w+4", "p_w+5"]] = df_grupo_test[
        ["p_w+1", "p_w+2", "p_w+3", "p_w+4", "p_w+5"]
    ]

df_pesos_df3 = pd.DataFrame(pesos_df3)

benchmark_col = "same_hour_mean_3w"

resultados_erros_df3 = []

for h in range(1, 6):
    real_col = f"w+{h}"
    pred_col = f"p_w+{h}"
    horizonte = f"w+{h}"

    df_eval = df3_test.dropna(subset=[real_col, pred_col, benchmark_col]).copy()

    y_real = df_eval[real_col]
    y_benchmark = df_eval[benchmark_col]
    y_model = df_eval[pred_col]

    mae_benchmark = mean_absolute_error(y_real, y_benchmark)
    mse_benchmark = mean_squared_error(y_real, y_benchmark)
    rmse_benchmark = np.sqrt(mse_benchmark)

    mae_model = mean_absolute_error(y_real, y_model)
    mse_model = mean_squared_error(y_real, y_model)
    rmse_model = np.sqrt(mse_model)

    resultados_erros_df3.append({
        "horizonte": horizonte,
        "MAE_benchmark": mae_benchmark,
        "MSE_benchmark": mse_benchmark,
        "RMSE_benchmark": rmse_benchmark,
        "MAE_model": mae_model,
        "MSE_model": mse_model,
        "RMSE_model": rmse_model,
        "diff MAE": mae_model - mae_benchmark,
        "diff MSE": mse_model - mse_benchmark,
        "diff RMSE": rmse_model - rmse_benchmark
    })

df_erros_df3 = pd.DataFrame(resultados_erros_df3)

print("\nResumo dos erros do df3:")
print(df_erros_df3)

print("\nPesos do modelo por idservico, setor e horizonte:")
print(df_pesos_df3)

cols_to_show = keys + [
    "consultas",
    "w-4", "w-3", "w-2", "w-1",
    "same_hour_mean_3w",
    "w+1", "p_w+1",
    "w+2", "p_w+2",
    "w+3", "p_w+3",
    "w+4", "p_w+4",
    "w+5", "p_w+5"
]

print("\nExemplo do dataframe final df3:")
print(df3_test[cols_to_show].head())

df3_resultado = df3_test.copy()

"""Benchmark 5 - XGBoost"""

df5 = df_consultas.copy()
df5["data"] = pd.to_datetime(df5["data"])

df5 = df5.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

keys = ["idservico", "setor", "data", "hora"]

base = df5[keys + ["consultas"]].copy()

def add_week(df5, base, col_name, days):
    ref = base.copy()
    ref["data"] = ref["data"] + timedelta(days=days)
    ref = ref.rename(columns={"consultas": col_name})
    return df5.merge(
        ref[keys + [col_name]],
        on=keys,
        how="left"
    )

df5 = add_week(df5, base, "w-4", 28)
df5 = add_week(df5, base, "w-3", 21)
df5 = add_week(df5, base, "w-2", 14)
df5 = add_week(df5, base, "w-1", 7)

df5 = add_week(df5, base, "w+1", -7)
df5 = add_week(df5, base, "w+2", -14)
df5 = add_week(df5, base, "w+3", -21)
df5 = add_week(df5, base, "w+4", -28)
df5 = add_week(df5, base, "w+5", -35)

df5 = df5.sort_values(by=["idservico", "setor", "data", "hora"]).reset_index(drop=True)

grp = df5.groupby(["idservico", "setor"])["consultas"]

df5["lag_24h"] = grp.shift(24 * 7 + 24)
df5["lag_48h"] = grp.shift(24 * 7 + 48)
df5["lag_72h"] = grp.shift(24 * 7 + 72)

df5["lag_1w"] = grp.shift(24 * 7)
df5["lag_2w"] = grp.shift(24 * 7 * 2)
df5["lag_3w"] = grp.shift(24 * 7 * 3)
df5["lag_4w"] = grp.shift(24 * 7 * 4)

df5["roll_mean_3h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(3).mean())
df5["roll_mean_6h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(6).mean())
df5["roll_mean_24h"] = grp.transform(lambda s: s.shift(24 * 7 + 1).rolling(24).mean())

df5["same_hour_mean_2w"] = df5[["lag_1w", "lag_2w"]].mean(axis=1)
df5["same_hour_mean_3w"] = df5[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df5["same_hour_mean_4w"] = df5[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df5["same_hour_wavg_4w"] = (0.4 * df5["lag_1w"] + 0.3 * df5["lag_2w"] + 0.2 * df5["lag_3w"] + 0.1 * df5["lag_4w"])

feature_cols = ["consultas","w-1", "w-2", "w-3", "w-4","lag_24h", "lag_48h", "lag_72h","lag_1w",
                "lag_2w", "lag_3w", "lag_4w","roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w",
                "same_hour_mean_3w", "same_hour_mean_4w","same_hour_wavg_4w"]

target_cols = ["w+1", "w+2", "w+3", "w+4", "w+5"]

needed_cols = feature_cols + target_cols

df5 = df5.dropna(subset=needed_cols).reset_index(drop=True)

df5_train = df5[(df5["data"] >= "2021-01-01") & (df5["data"] < "2025-01-01")].copy()
df5_test = df5[(df5["data"] >= "2025-01-01") & (df5["data"] < "2026-01-01")].copy()

modelos_xgb_df5 = {}

for h in range(1, 6):
    target_col = f"w+{h}"
    pred_col = f"p_w+{h}"

    X_train = df5_train[feature_cols]
    y_train = df5_train[target_col]

    X_test = df5_test[feature_cols]

    modelo = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        reg_alpha=0.5,
        random_state=42,
        tree_method="hist"
    )

    modelo.fit(X_train, y_train)

    df5_test[pred_col] = modelo.predict(X_test)
    df5_test[pred_col] = df5_test[pred_col].clip(lower=0)

    modelos_xgb_df5[target_col] = modelo

benchmark_col = "same_hour_mean_3w"

resultados_erros_df5 = []

for h in range(1, 6):
    real_col = f"w+{h}"
    pred_col = f"p_w+{h}"
    horizonte = f"w+{h}"

    y_real = df5_test[real_col]
    y_benchmark = df5_test[benchmark_col]
    y_model = df5_test[pred_col]

    mae_benchmark = mean_absolute_error(y_real, y_benchmark)
    mse_benchmark = mean_squared_error(y_real, y_benchmark)
    rmse_benchmark = np.sqrt(mse_benchmark)

    mae_model = mean_absolute_error(y_real, y_model)
    mse_model = mean_squared_error(y_real, y_model)
    rmse_model = np.sqrt(mse_model)

    resultados_erros_df5.append({
        "horizonte": horizonte,
        "MAE_benchmark": mae_benchmark,
        "MSE_benchmark": mse_benchmark,
        "RMSE_benchmark": rmse_benchmark,
        "MAE_model": mae_model,
        "MSE_model": mse_model,
        "RMSE_model": rmse_model,
        "diff MAE": mae_model - mae_benchmark,
        "diff MSE": mse_model - mse_benchmark,
        "diff RMSE": rmse_model - rmse_benchmark
    })

df_erros_df5 = pd.DataFrame(resultados_erros_df5)

print("\nResumo dos erros XGBoost vs benchmark:")
print(df_erros_df5)

cols_to_show = keys + [
    "consultas",
    "w-4", "w-3", "w-2", "w-1",
    "same_hour_mean_3w",
    "w+1", "p_w+1",
    "w+2", "p_w+2",
    "w+3", "p_w+3",
    "w+4", "p_w+4",
    "w+5", "p_w+5"
]

print("\nExemplo do dataframe final df5_test:")
print(df5_test[cols_to_show].head())

df5_resultado = df5_test.copy()

"""# bi benchmarks

## fixed basic
"""

df_eval = df_consultas.copy()
df_eval["data"] = pd.to_datetime(df_eval["data"])

df_eval = df_eval[["idservico", "data", "hora", "consultas","unidade", "setor", "servico", "cliente"]].copy()

def add_week(df, col_name, delta_days):
    ref = df[["idservico", "hora", "data", "consultas"]].copy()
    ref["data"] = ref["data"] + pd.Timedelta(days=delta_days)
    ref = ref.rename(columns={"consultas": col_name})

    return df.merge(ref[["idservico", "hora", "data", col_name]],on=["idservico", "hora", "data"],how="left")

df_eval = add_week(df_eval, "w-1", -7)
df_eval = add_week(df_eval, "w-2", -14)

for i in range(1, 6):
    df_eval = add_week(df_eval, f"w+{i}", 7 * i)

df_eval = df_eval.dropna(subset=["w-1", "w-2", "w+1", "w+2", "w+3", "w+4", "w+5"]).reset_index(drop=True)

df_eval["p_w+1_bench"] = df_eval[["consultas", "w-1", "w-2"]].mean(axis=1)
df_eval["p_w+2_bench"] = df_eval[["consultas", "w-1", "p_w+1_bench"]].mean(axis=1)
df_eval["p_w+3_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench"]].mean(axis=1)
df_eval["p_w+4_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench"]].mean(axis=1)
df_eval["p_w+5_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench"]].mean(axis=1)

df_eval = df_eval.sort_values(["idservico", "setor", "data", "hora"]).reset_index(drop=True)
grp = df_eval.groupby(["idservico", "setor"])["consultas"]


df_eval["lag_1w"] = grp.shift(24 * 7)
df_eval["lag_2w"] = grp.shift(24 * 7 * 2)
df_eval["lag_3w"] = grp.shift(24 * 7 * 3)
df_eval["lag_4w"] = grp.shift(24 * 7 * 4)

# Short-term lags anchored at w-1
df_eval["lag_24h"] = grp.shift(24*7 + 24)   # 24h before w-1
df_eval["lag_48h"] = grp.shift(24*7 + 48)   # 48h before w-1
df_eval["lag_72h"] = grp.shift(24*7 + 72)   # 72h before w-1

# Rolling means anchored at w-1
df_eval["roll_mean_3h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(3).mean())
df_eval["roll_mean_6h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(6).mean())
df_eval["roll_mean_24h"] = grp.transform(lambda s: s.shift(24*7 + 1).rolling(24).mean())

df_eval["same_hour_mean_2w"] = df_eval[["lag_1w", "lag_2w"]].mean(axis=1)
df_eval["same_hour_mean_3w"] = df_eval[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df_eval["same_hour_mean_4w"] = df_eval[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df_eval["same_hour_wavg_4w"] = (0.4 * df_eval["lag_1w"] +0.3 * df_eval["lag_2w"] +0.2 * df_eval["lag_3w"] +0.1 * df_eval["lag_4w"])

df_eval["idservico"] = df_eval["idservico"].astype("category")
df_eval["setor"] = df_eval["setor"].astype("category")

df_eval["idservico_cat"] = df_eval["idservico"].cat.codes
df_eval["setor_cat"] = df_eval["setor"].cat.codes

df_eval["Weekday"] = df_eval["data"].dt.weekday
df_eval["Month"] = df_eval["data"].dt.month
df_eval["day"] = df_eval["data"].dt.day

features = ["idservico_cat", "setor_cat", "hora", "Weekday", "Month", "day","lag_1w", "lag_2w", "lag_3w", "lag_4w","lag_24h", "lag_48h", "lag_72h",
    "roll_mean_3h", "roll_mean_6h", "roll_mean_24h","same_hour_mean_2w", "same_hour_mean_3w", "same_hour_mean_4w", "same_hour_wavg_4w"]

needed_cols = features + ["w+1", "w+2", "w+3", "w+4", "w+5","p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench", "p_w+5_bench"]

df_eval = df_eval.dropna(subset=needed_cols).reset_index(drop=True)

df_train = df_eval[(df_eval["data"] >= "2021-01-01") & (df_eval["data"] < "2024-07-01")].copy()
df_test  = df_eval[(df_eval["data"] >= "2025-01-01") & (df_eval["data"] < "2026-01-01")].copy()

targets = {"w+1": "w+1","w+2": "w+2","w+3": "w+3","w+4": "w+4","w+5": "w+5",}

models = {}

for horizon, target_col in targets.items():
    X_train = df_train[features]
    y_train = df_train[target_col]

    X_test = df_test[features]
    y_test = df_test[target_col]

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    pred_col = f"p_{horizon}_xgb"
    df_test[pred_col] = model.predict(X_test)
    df_test[pred_col] = np.clip(df_test[pred_col], 0, None)

    models[horizon] = model
    print(f"{horizon} done")

horizontes = ["w+1", "w+2", "w+3", "w+4", "w+5"]

resultados = []

for h in horizontes:
    y_true = df_test[h]

    y_pred_bench = df_test[f"p_{h}_bench"]
    mae_bench = mean_absolute_error(y_true, y_pred_bench)
    mse_bench = mean_squared_error(y_true, y_pred_bench)
    rmse_bench = np.sqrt(mse_bench)

    y_pred_xgb = df_test[f"p_{h}_xgb"]
    mae_xgb = mean_absolute_error(y_true, y_pred_xgb)
    mse_xgb = mean_squared_error(y_true, y_pred_xgb)
    rmse_xgb = np.sqrt(mse_xgb)

    resultados.append({
        "horizonte": h,
        "MAE_benchmark": mae_bench,
        "MSE_benchmark": mse_bench,
        "RMSE_benchmark": rmse_bench,
        "MAE_model": mae_xgb,
        "MSE_model": mse_xgb,
        "RMSE_model": rmse_xgb,
        "diff_MAE": mae_xgb - mae_bench,
        "diff_MSE": mse_xgb - mse_bench,
        "diff_RMSE": rmse_xgb - rmse_bench,
    })

resultado_metricas = pd.DataFrame(resultados)
print(resultado_metricas)

"""## linear regression"""

df_eval = df_consultas.copy()
df_eval["data"] = pd.to_datetime(df_eval["data"])

df_eval = df_eval[["idservico", "data", "hora", "consultas","unidade", "setor", "servico", "cliente"]].copy()

def add_week(df, col_name, delta_days):
    ref = df[["idservico", "hora", "data", "consultas"]].copy()
    ref["data"] = ref["data"] + pd.Timedelta(days=delta_days)
    ref = ref.rename(columns={"consultas": col_name})
    return df.merge(ref[["idservico", "hora", "data", col_name]],on=["idservico", "hora", "data"],how="left")

df_eval = add_week(df_eval, "w-1", -7)
df_eval = add_week(df_eval, "w-2", -14)

for i in range(1, 6):
    df_eval = add_week(df_eval, f"w+{i}", 7 * i)

df_eval = df_eval.dropna(subset=["w-1", "w-2", "w+1", "w+2", "w+3", "w+4", "w+5"]).reset_index(drop=True)

df_eval["p_w+1_bench"] = df_eval[["consultas", "w-1", "w-2"]].mean(axis=1)
df_eval["p_w+2_bench"] = df_eval[["consultas", "w-1", "p_w+1_bench"]].mean(axis=1)
df_eval["p_w+3_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench"]].mean(axis=1)
df_eval["p_w+4_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench"]].mean(axis=1)
df_eval["p_w+5_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench"]].mean(axis=1)

df_eval = df_eval.sort_values(["idservico", "setor", "data", "hora"]).reset_index(drop=True)
grp = df_eval.groupby(["idservico", "setor"])["consultas"]

# Fixed: anchored at w-1
df_eval["lag_24h"] = grp.shift(24*7 + 24)
df_eval["lag_48h"] = grp.shift(24*7 + 48)
df_eval["lag_72h"] = grp.shift(24*7 + 72)

df_eval["lag_1w"] = grp.shift(24 * 7)
df_eval["lag_2w"] = grp.shift(24 * 7 * 2)
df_eval["lag_3w"] = grp.shift(24 * 7 * 3)
df_eval["lag_4w"] = grp.shift(24 * 7 * 4)

# Fixed: rolling means anchored at w-1
df_eval["roll_mean_3h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(3).mean())
df_eval["roll_mean_6h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(6).mean())
df_eval["roll_mean_24h"] = grp.transform(lambda s: s.shift(24*7 + 1).rolling(24).mean())

df_eval["same_hour_mean_2w"] = df_eval[["lag_1w", "lag_2w"]].mean(axis=1)
df_eval["same_hour_mean_3w"] = df_eval[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df_eval["same_hour_mean_4w"] = df_eval[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df_eval["same_hour_wavg_4w"] = (0.4 * df_eval["lag_1w"] + 0.3 * df_eval["lag_2w"] + 0.2 * df_eval["lag_3w"] + 0.1 * df_eval["lag_4w"])

df_eval["idservico"] = df_eval["idservico"].astype("category")
df_eval["setor"] = df_eval["setor"].astype("category")

df_eval["idservico_cat"] = df_eval["idservico"].cat.codes
df_eval["setor_cat"] = df_eval["setor"].cat.codes

df_eval["Weekday"] = df_eval["data"].dt.weekday
df_eval["Month"] = df_eval["data"].dt.month
df_eval["day"] = df_eval["data"].dt.day

features = ["idservico_cat", "setor_cat", "hora", "Weekday", "Month", "day",
            "lag_1w", "lag_2w", "lag_3w", "lag_4w",
            "lag_24h", "lag_48h", "lag_72h",
            "roll_mean_3h", "roll_mean_6h", "roll_mean_24h",
            "same_hour_mean_2w", "same_hour_mean_3w", "same_hour_mean_4w", "same_hour_wavg_4w"]

features_lr = ["lag_1w", "lag_2w", "lag_3w", "lag_4w"]

needed_cols = features + ["w+1", "w+2", "w+3", "w+4", "w+5",
                          "p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench", "p_w+5_bench"]

df_eval = df_eval.dropna(subset=needed_cols).reset_index(drop=True)

df_train = df_eval[(df_eval["data"] >= "2021-01-01") & (df_eval["data"] < "2024-07-01")].copy()
df_test  = df_eval[(df_eval["data"] >= "2025-01-01") & (df_eval["data"] < "2026-01-01")].copy()

targets = {"w+1": "w+1", "w+2": "w+2", "w+3": "w+3", "w+4": "w+4", "w+5": "w+5"}

# --- Linear Regression benchmark (no intercept, 4 weekly lags) ---
from sklearn.linear_model import LinearRegression

lr_models = {}

for horizon, target_col in targets.items():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(df_train[features_lr], df_train[target_col])

    pred_col = f"p_{horizon}_lr"
    df_test[pred_col] = lr.predict(df_test[features_lr])
    df_test[pred_col] = np.clip(df_test[pred_col], 0, None)

    lr_models[horizon] = lr
    print(f"{horizon} LR done — coefs: {dict(zip(features_lr, lr.coef_.round(3)))}")

# --- XGBoost ---
models = {}

for horizon, target_col in targets.items():
    X_train = df_train[features]
    y_train = df_train[target_col]

    X_test = df_test[features]
    y_test = df_test[target_col]

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    pred_col = f"p_{horizon}_xgb"
    df_test[pred_col] = model.predict(X_test)
    df_test[pred_col] = np.clip(df_test[pred_col], 0, None)

    models[horizon] = model
    print(f"{horizon} XGB done")

# --- Comparison ---
horizontes = ["w+1", "w+2", "w+3", "w+4", "w+5"]
resultados = []

for h in horizontes:
    y_true = df_test[h]

    def metrics(y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    mae_bench,  mse_bench,  rmse_bench  = metrics(df_test[f"p_{h}_bench"])
    mae_lr,     mse_lr,     rmse_lr     = metrics(df_test[f"p_{h}_lr"])
    mae_xgb,    mse_xgb,    rmse_xgb    = metrics(df_test[f"p_{h}_xgb"])

    resultados.append({
        "horizonte":      h,
        "MAE_bench":      mae_bench,
        "MAE_lr":         mae_lr,
        "MAE_xgb":        mae_xgb,
        "MSE_bench":      mse_bench,
        "MSE_lr":         mse_lr,
        "MSE_xgb":        mse_xgb,
        "RMSE_bench":     rmse_bench,
        "RMSE_lr":        rmse_lr,
        "RMSE_xgb":       rmse_xgb,
        "diff_MAE_lr":    mae_lr  - mae_bench,
        "diff_MAE_xgb":   mae_xgb - mae_bench,
    })

resultado_metricas = pd.DataFrame(resultados)
print(resultado_metricas)

df_eval = df_consultas.copy()
df_eval["data"] = pd.to_datetime(df_eval["data"])

df_eval = df_eval[["idservico", "data", "hora", "consultas","unidade", "setor", "servico", "cliente"]].copy()

def add_week(df, col_name, delta_days):
    ref = df[["idservico", "setor", "hora", "data", "consultas"]].copy()
    ref["data"] = ref["data"] + pd.Timedelta(days=delta_days)
    ref = ref.rename(columns={"consultas": col_name})

    return df.merge(
        ref[["idservico", "setor", "hora", "data", col_name]],
        on=["idservico", "setor", "hora", "data"],
        how="left"
    )

df_eval = add_week(df_eval, "w-1", 7)
df_eval = add_week(df_eval, "w-2", 14)

for i in range(1, 6):
    df_eval = add_week(df_eval, f"w+{i}", -7 * i)

df_eval = df_eval.dropna(subset=["w-1", "w-2", "w+1", "w+2", "w+3", "w+4", "w+5"]).reset_index(drop=True)

df_eval["p_w+1_bench"] = df_eval[["consultas", "w-1", "w-2"]].mean(axis=1)
df_eval["p_w+2_bench"] = df_eval[["consultas", "w-1", "p_w+1_bench"]].mean(axis=1)
df_eval["p_w+3_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench"]].mean(axis=1)
df_eval["p_w+4_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench"]].mean(axis=1)
df_eval["p_w+5_bench"] = df_eval[["consultas", "p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench"]].mean(axis=1)

df_eval = df_eval.sort_values(["idservico", "setor", "data", "hora"]).reset_index(drop=True)
grp = df_eval.groupby(["idservico", "setor"])["consultas"]

# Fixed: anchored at w-1
df_eval["lag_24h"] = grp.shift(24*7 + 24)
df_eval["lag_48h"] = grp.shift(24*7 + 48)
df_eval["lag_72h"] = grp.shift(24*7 + 72)

df_eval["lag_1w"] = grp.shift(24 * 7)
df_eval["lag_2w"] = grp.shift(24 * 7 * 2)
df_eval["lag_3w"] = grp.shift(24 * 7 * 3)
df_eval["lag_4w"] = grp.shift(24 * 7 * 4)

# Fixed: rolling means anchored at w-1
df_eval["roll_mean_3h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(3).mean())
df_eval["roll_mean_6h"]  = grp.transform(lambda s: s.shift(24*7 + 1).rolling(6).mean())
df_eval["roll_mean_24h"] = grp.transform(lambda s: s.shift(24*7 + 1).rolling(24).mean())

df_eval["same_hour_mean_2w"] = df_eval[["lag_1w", "lag_2w"]].mean(axis=1)
df_eval["same_hour_mean_3w"] = df_eval[["lag_1w", "lag_2w", "lag_3w"]].mean(axis=1)
df_eval["same_hour_mean_4w"] = df_eval[["lag_1w", "lag_2w", "lag_3w", "lag_4w"]].mean(axis=1)

df_eval["same_hour_wavg_4w"] = (0.4 * df_eval["lag_1w"] + 0.3 * df_eval["lag_2w"] + 0.2 * df_eval["lag_3w"] + 0.1 * df_eval["lag_4w"])

df_eval["idservico"] = df_eval["idservico"].astype("category")
df_eval["setor"] = df_eval["setor"].astype("category")

df_eval["idservico_cat"] = df_eval["idservico"].cat.codes
df_eval["setor_cat"] = df_eval["setor"].cat.codes

df_eval["Weekday"] = df_eval["data"].dt.weekday
df_eval["Month"] = df_eval["data"].dt.month
df_eval["day"] = df_eval["data"].dt.day

features = ["idservico_cat", "setor_cat", "hora", "Weekday", "Month", "day",
            "lag_1w", "lag_2w", "lag_3w", "lag_4w",
            "lag_24h", "lag_48h", "lag_72h",
            "roll_mean_3h", "roll_mean_6h", "roll_mean_24h",
            "same_hour_mean_2w", "same_hour_mean_3w", "same_hour_mean_4w", "same_hour_wavg_4w"]

features_lr = ["lag_1w", "lag_2w", "lag_3w", "lag_4w"]

needed_cols = features + ["w+1", "w+2", "w+3", "w+4", "w+5",
                          "p_w+1_bench", "p_w+2_bench", "p_w+3_bench", "p_w+4_bench", "p_w+5_bench"]

df_eval = df_eval.dropna(subset=needed_cols).reset_index(drop=True)

df_train = df_eval[(df_eval["data"] >= "2021-01-01") & (df_eval["data"] < "2024-07-01")].copy()
df_test  = df_eval[(df_eval["data"] >= "2025-01-01") & (df_eval["data"] < "2026-01-01")].copy()

targets = {"w+1": "w+1", "w+2": "w+2", "w+3": "w+3", "w+4": "w+4", "w+5": "w+5"}

# --- Linear Regression benchmark (no intercept, 4 weekly lags) ---
from sklearn.linear_model import LinearRegression

lr_models = {}

for horizon, target_col in targets.items():
    lr = LinearRegression(fit_intercept=False)
    lr.fit(df_train[features_lr], df_train[target_col])

    pred_col = f"p_{horizon}_lr"
    df_test[pred_col] = lr.predict(df_test[features_lr])
    df_test[pred_col] = np.clip(df_test[pred_col], 0, None)

    lr_models[horizon] = lr
    print(f"{horizon} LR done — coefs: {dict(zip(features_lr, lr.coef_.round(3)))}")

# --- XGBoost ---
models = {}

for horizon, target_col in targets.items():
    X_train = df_train[features]
    y_train = df_train[target_col]

    X_test = df_test[features]
    y_test = df_test[target_col]

    model = XGBRegressor(
        objective="count:poisson",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    pred_col = f"p_{horizon}_xgb"
    df_test[pred_col] = model.predict(X_test)
    df_test[pred_col] = np.clip(df_test[pred_col], 0, None)

    models[horizon] = model
    print(f"{horizon} XGB done")

# --- Comparison ---
horizontes = ["w+1", "w+2", "w+3", "w+4", "w+5"]
resultados = []

for h in horizontes:
    y_true = df_test[h]

    def metrics(y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mae, mse, rmse

    mae_bench,  mse_bench,  rmse_bench  = metrics(df_test[f"p_{h}_bench"])
    mae_lr,     mse_lr,     rmse_lr     = metrics(df_test[f"p_{h}_lr"])
    mae_xgb,    mse_xgb,    rmse_xgb    = metrics(df_test[f"p_{h}_xgb"])

    resultados.append({
        "horizonte":      h,
        "MAE_bench":      mae_bench,
        "MAE_lr":         mae_lr,
        "MAE_xgb":        mae_xgb,
        "MSE_bench":      mse_bench,
        "MSE_lr":         mse_lr,
        "MSE_xgb":        mse_xgb,
        "RMSE_bench":     rmse_bench,
        "RMSE_lr":        rmse_lr,
        "RMSE_xgb":       rmse_xgb,
        "diff_MAE_lr":    mae_lr  - mae_bench,
        "diff_MAE_xgb":   mae_xgb - mae_bench,
    })

resultado_metricas = pd.DataFrame(resultados)
print(resultado_metricas)

