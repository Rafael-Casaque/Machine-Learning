import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px

base_credito = pd.read_csv('Data-Science-and-Machine-Learning\Data-Pre-Processing\credit_data.csv') #faz a leitura do arquivo base
print(base_credito.head(10)) #exibe os 10 primeiros registros
print(base_credito.tail(10)) #exibe os 10 últimos registros
print(base_credito.describe()) #exibe a contagem para cada um dos atributos numéricos
print(base_credito[base_credito['person_income'] >=1000000]) #retorna as pessoas com renda anual >= $1.000.000,00
print(np.unique(base_credito['cb_person_default_on_file'], return_counts=True)) #valores/qp existentes em cb_person_default_on_file
# translate atributes en/pt-br

# person_age                    =>      Idade_Pessoa
# person_income                 =>      Renda_Pesoa
# person_home_ownership         =>      Situação_Casa_Pesoa
# person_emp_length             =>      Emprego_Pessoa
# loan_intent                   =>      Intenção_Empréstimo
# loan_grade                    =>      Grau_Empréstimo
# loan_amnt                     =>      Montante_Empréstimo
# loan_int_rate                 =>      Taxa_Juros_Empréstimo
# loan_status                   =>      Status_Empréstimo
# loan_percent_income           =>      Renda_Percentual_Empréstimo
# cb_person_default_on_file     =>      Pessoa_Pagou_Anteriormente
# cb_person_cred_hist_length    =>      Histórico_Cumprimento_Empréstimo
