import pandas as pd
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

base_credito = pd.read_csv('Data pre-processing\credit_data.csv') #realiza a importação do banco de dados

#tratamento de valores inconsistentes 

#1ª forma: deletando a coluna inteira

base_credito2 = base_credito.drop('person_age', axis=1) #comando para apagar a coluna inteira (axis = 0 linha, 1 coluna)
print(base_credito2) #exibe o banco de dados - 1 coluna

#2ª forma: deletando a linha inteira

base_credito3 = base_credito.drop(base_credito[base_credito['person_age']>99].index) #comando para apagar as linhas dos usuários com age>99
print(base_credito3.loc[base_credito3['person_age']>99]) #exibe o banco de dados - 5 linhas

#3ª forma: preenchendo manualmente
#exemplo: através do contato, ligar manualmente, perguntar a idade e alterar diretamente no banco de dados 

#4ªforma: preencher com a média dos valores

print(base_credito3.mean()) #retorna a média de todos os atributos
print('\n')
print(base_credito3['person_age'].mean()) #retorna a média somente da idade
print('\n')
print(base_credito['person_age'] [base_credito['person_age']<99].mean()) #retorna a média somente das idades < 99

base_credito.loc[base_credito['person_age']>99, 'person_age'] = 27.71804 #atualiza os valores inconsistentes com o valor da média somente do campo 'person_age

print(base_credito.loc[81]) #exibe os atributos do id 81

grafico = px.scatter_matrix(base_credito,dimensions=['person_age', 'person_income'], color ='cb_person_default_on_file' )
grafico.show()