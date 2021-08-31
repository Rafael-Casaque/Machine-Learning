import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px

base_censo = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\census_data.csv') #faz a leitura do arquivo base

print(np.unique(base_censo['income'])); # retorna os valores existentes em income
print(np.unique(base_censo['income'], return_counts=True)) # retorna a quantidade de valores em income
sb.countplot(x = base_censo['income']) # cria o gráfico em que x = income e y = counts
plt.show() # exibe o gráfico 
plt.hist(x = base_censo['age']); #cria um gráfico categórico para idades
plt.show() # exibe o gráfico 
plt.hist(x = base_censo['education-num']); #cria um gráfico categórico para anos de estudo
plt.show() # exibe o gráfico 
plt.hist(x = base_censo['hour-per-week']); #cria um gráfico categórico para horas trabalhadas semanalmente
plt.show() # exibe o gráfico 
grafico = px.treemap(base_censo, path=['workclass']) #cria um gráfico dinâmico por agrupamento para setor de trabalho
grafico.show() #exibe o gráfico no browser
grafico2 = px.treemap(base_censo, path=['workclass','age']) #cria um gráfico dinâmico por agrupamento para setor de trabalho e idades
grafico2.show() #exibe o gráfico no browser
grafico3 = px.treemap(base_censo, path=['occupation','relationship']) #cria um gráfico dinâmico por agrupamento para ocupação e relacionamento
grafico3.show() #exibe o gráfico no browser
grafico4 = px.treemap(base_censo, path=['occupation','relationship','age']) #cria um gráfico dinâmico por agrupamento para ocupação e relacionamento e idades
grafico4.show() #exibe o gráfico no browser
grafico5 = px.parallel_categories(base_censo, dimensions=['occupation','relationship']) #cria um gráfico com linhas paralelas que conectam ocupação com relacionamento
grafico5.show() #exibe o gráfico no browser
grafico6 = px.parallel_categories(base_censo, dimensions=['workclass','occupation','income']) #cria um gráfico com linhas paralelas que conectam setor de trabalho, ocupação e renda anual
grafico6.show() #exibe o gráfico no browser
grafico7 = px.parallel_categories(base_censo, dimensions=['education','income']) #cria um gráfico com linhas paralelas que conectam o nível de educação com renda anual
grafico7.show() #exibe o gráfico no browser

