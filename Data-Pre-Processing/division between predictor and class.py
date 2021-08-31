import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['age']<0, 'age'] = 40.92770044906149 #atualiza os valores inconsistentes com o valor da média somente do campo 'age
base_credito['age'].fillna(base_credito['age'].mean(), inplace=True) #substitui os valores vazios de age pela média 
print(base_credito.loc[2])
x_credito = base_credito.iloc[:,1:4].values #atribui À variável x_credito apenas as colunas de 1-3 (:) indica todas as linhas
print(x_credito) #os valores ficam armazenados como arrays, necessários para as bibliotecas de leitura (convertida pra numpy.ndarray)
y_credito = base_credito.iloc[:,4].values #atribui À variável y_credito apenas a colunas 4
print(y_credito)
