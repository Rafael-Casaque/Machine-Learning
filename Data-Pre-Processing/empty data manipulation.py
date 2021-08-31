import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['age']<0, 'age'] = 40.92770044906149 #atualiza os valores inconsistentes com o valor da média somente do campo 'person_age

print(base_credito.isnull()) #retorna false para valores preenchidos e true para valores vazios
print(base_credito.isnull().sum()) #retorna a soma dos valores que estão vazios
print(base_credito.loc[pd.isnull(base_credito['age'])]) #retorna os usuários sem valor para age
base_credito['age'].fillna(base_credito['age'].mean(), inplace=True) #substitui os valores vazios de age pela média 
#base_credito.loc[base_credito['person_home_ownership']=='OWN', 'person_home_ownership'] = 'PRÓPRIA' #substituir uma string por outra
print(base_credito.isnull().sum())




