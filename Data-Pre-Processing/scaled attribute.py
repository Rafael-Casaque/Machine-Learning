import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 
from sklearn.preprocessing import StandardScaler


base_credito = pd.read_csv('Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['person_age']>99, 'person_age'] = 27.71804 #atribui a media às idades>99
base_credito['loan_int_rate'].fillna(base_credito['loan_int_rate'].mean(), inplace=True) #substitui os valores vazios de loan_rate_int pela média 
base_credito['person_emp_length'].fillna(base_credito['person_emp_length'].mean(), inplace=True) #substitui os valores vazios de person_emp_length pela média 
x_credito = base_credito.iloc[:,0:10].values #atribui À variável x_credito apenas as colunas de 0-9 (:) indica todas as linhas
y_credito = base_credito.iloc[:,10].values #atribui À variável y_credito apenas a colunas 11

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

escala_creito = StandardScaler()
#x_credito = escala_creito.fit_transform(x_credito[:,0:2]) com intervalo 
x_credito = escala_creito.fit_transform(x_credito[:,[0,1,3,6,7,8,9]]) #sem intervalo, coontando coluna por coluna
print(x_credito[:,0].min()) #exibe o menor valor
print(x_credito[:,0].max()) #exibe o maior valor
print(x_credito)


