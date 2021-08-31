import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px

base_censo = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\census_data.csv') #faz a leitura do arquivo base
print(base_censo.columns) #retorna as colunas da base de dados
x_censo = base_censo.iloc[:,0:14].values #atribui a variável os atributos preditores, todas as linhas colunas 0-13 . values para converter em numpy array
y_censo = base_censo.iloc[:,14].values #atribui a variável o atributo classe, todas as linhas coluna 14 . values para converter em numpy array

print(x_censo) # exibe os regristros de atributos preditores
print(y_censo) # exibe os regristros de atributo classe