import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['person_age']>99, 'person_age'] = 27.71804 #atribui a media às idades>99

print(base_credito.isnull()) #retorna false para valores preenchidos e true para valores vazios
print(base_credito.isnull().sum()) #retorna a soma dos valores que estão vazios
print(base_credito.loc[pd.isnull(base_credito['loan_int_rate'])]) #retorna os usuários sem valor para loan_int_rate 
base_credito['loan_int_rate'].fillna(base_credito['loan_int_rate'].mean(), inplace=True) #substitui os valores vazios de loan_rate_int pela média 
base_credito['person_emp_length'].fillna(base_credito['person_emp_length'].mean(), inplace=True) #substitui os valores vazios de person_emp_length pela média 
#base_credito.loc[base_credito['person_home_ownership']=='OWN', 'person_home_ownership'] = 'PRÓPRIA' #substituir uma string por outra






