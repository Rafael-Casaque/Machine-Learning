import pandas as pd
from pandas.core.dtypes.missing import isnull
from pandas.core.indexes import base
import plotly.express as px 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

#Base Crédito

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #realiza a importação do banco de dados
base_credito.loc[base_credito['age']<0, 'age'] = 40.92770044906149 #atualiza os valores inconsistentes com o valor da média somente do campo 'person_age
base_credito['age'].fillna(base_credito['age'].mean(), inplace=True) #substitui os valores vazios de age pela média 
x_credito = base_credito.iloc[:,1:4].values #atribui À variável x_credito apenas as colunas de 1-3 (:) indica todas as linhas
y_credito = base_credito.iloc[:,4].values #atribui À variável y_credito apenas a colunas 4
escala_credito = StandardScaler()
x_credito = escala_credito.fit_transform(x_credito) # deixa os atributos na mesma escala

#Base Censo

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
onehotencoder_censo = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
x_censo = onehotencoder_censo.fit_transform(x_censo).toarray()
escala_censo = StandardScaler()
x_censo = escala_censo.fit_transform(x_censo) # deixa todos os atributos na mesma escala
x_credito_treinamento, x_credito_teste, y_credito_treinamento, y_credito_teste = train_test_split(x_credito,y_credito, test_size=0.25, random_state=0) 
x_censo_treinamento, x_censo_teste, y_censo_treinamento, y_censo_teste = train_test_split(x_censo,y_censo, test_size=0.15, random_state=0) 

# Usado para salvar as variáveis em um arquivo

with open('Data-Science-and-Machine-Learning\Data-Pre-Processing\credito.pkl', mode='wb') as f:
    pickle.dump([x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste],f)

with open('Data-Science-and-Machine-Learning\Data-Pre-Processing\censo.pkl', mode='wb') as f:
    pickle.dump([x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste],f) 
