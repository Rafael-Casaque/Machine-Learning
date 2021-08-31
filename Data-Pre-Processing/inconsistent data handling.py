import pandas as pd
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados

#tratamento de valores inconsistentes 

#1ª forma: deletando a coluna inteira

base_credito2 = base_credito.drop('age', axis=1) #comando para apagar a coluna inteira (axis = 0 linha, 1 coluna)
print(base_credito2) #exibe o banco de dados - 1 coluna

#2ª forma: deletando a linha inteira

base_credito3 = base_credito.drop(base_credito[base_credito['age']<0].index) #comando para apagar as linhas dos usuários com age>99
print(base_credito3.loc[base_credito3['age']<0]) #exibe o banco de dados - 3 linhas

#3ª forma: preenchendo manualmente
#exemplo: através do contato, ligar manualmente, perguntar a idade e alterar diretamente no banco de dados 

#4ªforma: preencher com a média dos valores

print(base_credito3.mean()) #retorna a média de todos os atributos
print('\n')
print(base_credito3['age'].mean()) #retorna a média somente da idade
print('\n')
print(base_credito['age'] [base_credito['age']>0].mean()) #retorna a média somente das idades >0

base_credito.loc[base_credito['age']<0, 'age'] = 40.92770044906149 #atualiza os valores inconsistentes com o valor da média somente do campo 'person_age

print(base_credito.loc[15]) #exibe os atributos do id 15

grafico = px.scatter_matrix(base_credito,dimensions=['age', 'income'], color ='default' )
grafico.show()