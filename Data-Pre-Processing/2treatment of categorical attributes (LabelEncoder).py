import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

base_censo = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\census_data.csv') #faz a leitura do arquivo base
x_censo = base_censo.iloc[:,0:14].values #atribui a variável os atributos preditores, todas as linhas colunas 0-13 . values para converter em numpy array
y_censo = base_censo.iloc[:,14].values #atribui a variável o atributo classe, todas as linhas coluna 14 . values para converter em numpy array

# O trabamento de atributos catégoricos é feito em duas etapas, LabelEncoder e OneHotEncoder

# Esse arquivo se trata do LabelEncoder

# Não é possível realizar os cálculos matemáticos de machine learning utilizando strings
# Logo, é necessário realizar uma conversão para que todos sejam dados numéricos

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

print(x_censo[0]) #printa todos os atributos da linha 0, agora todos são números
