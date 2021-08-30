import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['person_age']>99, 'person_age'] = 27.71804 #atribui a media às idades>99
base_credito['loan_int_rate'].fillna(base_credito['loan_int_rate'].mean(), inplace=True) #substitui os valores vazios de loan_rate_int pela média 
base_credito['person_emp_length'].fillna(base_credito['person_emp_length'].mean(), inplace=True) #substitui os valores vazios de person_emp_length pela média 
print(base_credito.loc[2])
x_credito = base_credito.iloc[:,0:10].values #atribui À variável x_credito apenas as colunas de 0-9 (:) indica todas as linhas
print(x_credito) #os valores ficam armazenados como arrays, necessários para as bibliotecas de leitura (convertida pra numpy.ndarray)
y_credito = base_credito.iloc[:,10].values #atribui À variável y_credito apenas a colunas 11
print(y_credito)
