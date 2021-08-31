import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #faz a leitura do arquivo base
print(base_credito.head(10)) #exibe os 10 primeiros registros
print(base_credito.tail(10)) #exibe os 10 últimos registros
print(base_credito.describe()) #exibe a contagem para cada um dos atributos numéricos
print(base_credito[base_credito['income'] >=1000]) #retorna as pessoas com renda anual >= $1.000,00
print(np.unique(base_credito['default'], return_counts=True)) #valores/que pagaram existentes em default
# translate atributes en/pt-br

# clienteid     =>      Idade_Pessoa
# income        =>      Renda_Pesoa
# age           =>      Situação_Casa_Pesoa
# loan          =>      Emprego_Pessoa
# default       =>      Intenção_Empréstimo
