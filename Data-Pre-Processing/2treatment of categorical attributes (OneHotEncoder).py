import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

base_censo = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\census_data.csv') #faz a leitura do arquivo base
x_censo = base_censo.iloc[:,0:14].values #atribui a variável os atributos preditores, todas as linhas colunas 0-13 . values para converter em numpy array
y_censo = base_censo.iloc[:,14].values #atribui a variável o atributo classe, todas as linhas coluna 14 . values para converter em numpy array
label_encoder_workclass = LabelEncoder() #instancia a variável no objeto labelencoder
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupacion = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
x_censo[:,1] = label_encoder_workclass.fit_transform(x_censo[:,1]) #transforma as string de todas as linhas e a coluna referida em número e aloca em x_censo[coluna referida]
x_censo[:,3] = label_encoder_education.fit_transform(x_censo[:,3])
x_censo[:,5] = label_encoder_marital.fit_transform(x_censo[:,5])
x_censo[:,6] = label_encoder_occupacion.fit_transform(x_censo[:,6])
x_censo[:,7] = label_encoder_relationship.fit_transform(x_censo[:,7])
x_censo[:,8] = label_encoder_race.fit_transform(x_censo[:,8])
x_censo[:,9] = label_encoder_sex.fit_transform(x_censo[:,9])
x_censo[:,13] = label_encoder_country.fit_transform(x_censo[:,13])


# Esse arquivo se trata do OneHotEncoder

# O problema de se utilizar apenas o LabelEncoder é que teremos muitas categorias e valores diferentes
# Dessa forma, os algoritmos de machine Learning podem considerar uma categoria de índice 15 mais importante que a 1

#Exemplo: há três marcas de carros

# Gol, Pálio e Uno
#  1     2      3

#Nesse sentido, caso o algorítmo usasse uma multiplicação, o Uno teria maior peso
#Para resolver esse problema, pode ser feito uma colunização:

#Gol recebe     1 0 0   1 na primeira coluna
#Pálio recebe   0 1 0   1 na segunda coluna
#Uno recebe     0 0 1   1 na terceira coluna

print(np.unique(base_censo['workclass'])) #exibe os elementos em workclass

print(len(np.unique(base_censo['workclass']))) #exibe quantos elementos há em workclass

onehotencoder_censo = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')

#Transforma cada coluna indicada em várias outras colunas, assim como mostrado no exemplo anterior 
#O passtrought serve para que não apague os atributos númericos que não foram selecionados

x_censo = onehotencoder_censo.fit_transform(x_censo).toarray()

print(x_censo[0]) #exibe as informações da linha 0
print(x_censo.shape) # exibe a quantidade de linhas x colunas

# as colunas indicam a somatória de cada "subcoluna"