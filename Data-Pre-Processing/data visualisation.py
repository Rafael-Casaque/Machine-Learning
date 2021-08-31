import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #faz a leitura do arquivo base

sb.countplot(x = base_credito['age']); #cria um gráfico de barras com o índice no eixo x
plt.show() #faz a exibição do gráfico em uma janela

plt.hist( x = base_credito['age']); #cria um gráfico de barras com agrupamentos de intervalos
plt.show() #ao fechar a primeira janela, cria-se outra

plt.hist( x = base_credito['loan']);
plt.show()

grafico = px.scatter_matrix(base_credito, dimensions=['age','income']); #cria o gráfico 1º índice x 2º índice
grafico.show() #abre o grafico no browser

grafico = px.scatter_matrix(base_credito, dimensions=['age','income'],color='default');
grafico.show() #abre o grafico no browser