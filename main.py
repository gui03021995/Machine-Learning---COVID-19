
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re

df1 = pd.read_csv('HIST_PAINEL_COVIDBR_2022_Parte1_19out2022.csv', on_bad_lines='skip', sep=";")
df2 = pd.read_csv('HIST_PAINEL_COVIDBR_2022_Parte2_19out2022.csv', on_bad_lines='skip', sep=";")

df = pd.concat([df1,df2])
#brasil.dtypes
brasil = df.loc[(df.regiao == 'Brasil') & (df.casosAcumulado > 0)]

brasil = brasil.drop(columns=['estado', 'municipio', 'coduf', 'codmun','codRegiaoSaude', 'nomeRegiaoSaude',
                'interior/metropolitana', 'emAcompanhamentoNovos', 'casosNovos','obitosNovos']

brasil = brasil.rename(columns={"semanaEpi": "semana", "populacaoTCU2019":"populacao", "casosAcumulado":"casos", "obitosAcumulado":"obitos", "Recuperadosnovos": "recuperados"})

#brasil.dtypes

brasil["data"] = brasil["data"].astype("datetime64[ns]")

px.line(brasil, 'data', 'casos',
        labels={'data':'Data', 'casos':'Número de casos confirmados'},
       title='Casos confirmados no Brasil')

#função para fazer a contagem de novos casos
brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['casos'].iloc[x] - brasil['casos'].iloc[x-1],
    np.arange(brasil.shape[0])
))

# Visualizando
px.line(brasil, x='data', y='novoscasos', title='Novos casos por dia',
       labels={'data': 'Data', 'novoscasos': 'Novos casos'})


fig = go.Figure()

fig.add_trace(
    go.Scatter(x=brasil.data, y=brasil.obitos, name='Mortes', mode='lines+markers',
              line=dict(color='red')))

#Layout
fig.update_layout(title='Mortes por COVID-19 no Brasil',
                   xaxis_title='Data',
                   yaxis_title='Número de mortes')
fig.show()

#Função para taxa de crescimento
def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # Se data_inicio for None, define como a primeira data disponível no dataset
    if data_inicio == None:
        data_inicio = data.data.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    if data_fim == None:
        data_fim = data.data.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    # Define os valores de presente e passado
    passado = data.loc[data.data == data_inicio, variable].values[0]
    presente = data.loc[data.data == data_fim, variable].values[0]

    # Define o número de pontos no tempo q vamos avaliar
    n = (data_fim - data_inicio).days

    # Calcula a taxa
    taxa = (presente / passado) ** (1 / n) - 1

    return taxa * 100

cresc_medio = taxa_crescimento(brasil, 'casos')
print(f"O crescimento médio do COVID no Brasil no período avaliado foi de {cresc_medio.round(2)}%.")

#Agora, vamos observar o comportamento da taxa de crescimento no tempo. Para isso, vamos definir uma função para calcular a taxa de crescimento semanal.

def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.data.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    data_fim = data.data.max()
    n = (data_fim - data_inicio).days
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
        range(1, n + 1)
    ))
    return np.array(taxas) * 100

tx_dia = taxa_crescimento_diaria(brasil, 'casos')

#tx_dia

primeiro_dia = brasil.data.loc[brasil.casos > 0].min()
px.line(x=pd.date_range(primeiro_dia, brasil.data.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil',
       labels={'y':'Taxa de crescimento', 'x':'Data'})

-------------------------------------------------------------------------------------------------------------
##Predições

#Vamos construir um modelo de séries temporais para prever os novos casos. Antes analisemos a série temporal.

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

novoscasos = brasil.novoscasos
novoscasos.index = brasil.data

res = seasonal_decompose(novoscasos)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.scatter(novoscasos.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

#Decompondo a série de confirmados
confirmados = brasil.novoscasos
confirmados.index = brasil.data

res2 = seasonal_decompose(confirmados)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
ax1.plot(res2.observed)
ax2.plot(res2.trend)
ax3.plot(res2.seasonal)
ax4.scatter(confirmados.index, res2.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

#Predizendo o número de casos confirmados com um AUTO-ARIMA
!pip install pmdarima
from pmdarima.arima import auto_arima
!pip install pmdarima

modelo = auto_arima(confirmados)

pd.date_range('2020-10-01', '2022-10-19')

fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observed'
))

fig.add_trace(go.Scatter(x=confirmados.index, y = modelo.predict_in_sample(), name='Predicted'))
fig.add_trace(go.Scatter(x=pd.date_range('2022-10-20', '2022-11-05'), y=modelo.predict(15), name='Forecast'))
fig.update_layout(title='Previsão de casos confirmados para os próximos 16 dias',
                 yaxis_title='Casos confirmados', xaxis_title='Data')
fig.show()
-------------------------------------------------------------------------------------------------------------------------
#Forecasting com Facebook Prophet
!conda install -c conda-forge fbprophet -y
from fbprophet import Prophet

# preparando os dados
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# renomeia colunas
train.rename(columns={"data":"ds","novoscasos":"y"},inplace=True)
test.rename(columns={"data":"ds","novoscasos":"y"},inplace=True)
test = test.set_index("ds")
test = test['y']

profeta = Prophet(growth="logistic", changepoints=['2022-09-22', '2022-09-30', '2022-10-10', '2022-10-05', '2022-10-14'])

#pop = 1000000
pop = 211463256 #https://www.ibge.gov.br/apps/populacao/projecao/box_popclock.php
train['cap'] = pop

# Treina o modelo
profeta.fit(train)

# Construindo previsões para o futuro
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast =  profeta.predict(future_dates)

fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
fig.add_trace(go.Scatter(x=test.index, y=test, name='Observados - Teste'))
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.show()

