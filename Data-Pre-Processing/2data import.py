import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px

base_censo = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\census_data.csv') #faz a leitura do arquivo base
print(base_censo.describe()) #descrição dos atributos numéricos já inclui a média 
print(base_censo.isnull().sum()) #soma dos valores faltantes

# translate atributes en/pt-br

# age                       idade => variável número discreta 
# workclass                 setor de trabalho => variável categórica nominal 
# final-weight              cálculo do senso => variável número contínuo 
# education                 grau de estudo => variável categórica ordinal   
# education-num             anos de estudo => variável número discreta
# merital-status            estado civil => variável categórica nominal
# occupation                ocupação => variável categórica nominal
# relationship              relacionamento => variável categórica nominal
# race                      raça => variável categórica nominal
# sex                       sexo => variável categórica nominal
# capital-gain              ganho de capital => variável número contínua
# capital-loss              perda de capital => variável número contínua
# hour-per-week             horas trabalhadas por semana => variável número contínua
# native-country            país de origem => variável categórica nominal
# income                    salário anual => variável categórica ordinal (<50 e >50) será o atributo a ser previsisto


