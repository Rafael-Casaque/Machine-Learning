import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 
from sklearn.preprocessing import StandardScaler


base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['age']<0, 'age'] = 40.92770044906149 #atualiza os valores inconsistentes com o valor da média somente do campo 'person_age
base_credito['age'].fillna(base_credito['age'].mean(), inplace=True) #substitui os valores vazios de age pela média 
x_credito = base_credito.iloc[:,1:4].values #atribui À variável x_credito apenas as colunas de 1-3 (:) indica todas as linhas
y_credito = base_credito.iloc[:,4].values #atribui À variável y_credito apenas a colunas 4

print(x_credito[:,0]) #exibe todas as linhas apenas da coluna 0 'person_age'
print(x_credito[:,0].min()) #exibe o menor valor
print(x_credito[:,0].max()) #exibe o maior valor

# É importante deixar todos os valores na mesma escala, senão o algorítmo pode dar mais relevância a determinado atributo em detrimento de outro

# Formulas de fazer essa operação:

# Padronização (Standardisation) ===> x = x - média(x) / desvio padrão(x)
# Normalização (Normalization) ===> x = x - mínimo(x) / máximo(x) - mínimo(x)

# Desvio padrão == quanto os valores variam quando comparados à média

# A Padronização é mais indicada quando há registros muito fora do padrão (outliers)

#Funções para o processo de padronização

escala_credito = StandardScaler()
x_credito = escala_credito.fit_transform(x_credito) # deixa os atributos na mesma escala
print(x_credito[:,0].min()) #exibe o menor valor da coluna 0
print(x_credito[:,1].min()) #exibe o maior valor da coluna 1 
print(x_credito[:,2].min()) #exibe o maior valor da coluna 2 
print(x_credito)



