import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt


with open('Data-Science-and-Machine-Learning\Bayesian-Learning\credito.pkl', 'rb') as f: #carrega o arquivo pré-processado anteriormente
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f) #armazena as informações do arquivo nas variáveis indicadas

print(x_credito_treinamento.shape)

naive_dados_credito = GaussianNB()  #linhas para fazer a geração da tabela de probabilidade
naive_dados_credito.fit(x_credito_treinamento, y_credito_treinamento)   #o primeiro parâmetro a ser passado é sempre dos atributos previsores, depois da classe

# Com isso o algorítmo já está treinado e podemos fazer as previsões

previsoes = naive_dados_credito.predict(x_credito_teste) #linha usada para prever as classes do banco de dados x_credito_teste

print(previsoes) #exibe as previsões, sendo 1 que não paga o empréstimo e 0 que paga
print('\n')
print(y_credito_teste) #exibe as previsões reais para comparações e ver como o algorítmo acertou

# para fazer essa comparação podemos usar a seguinte função:

print(accuracy_score(y_credito_teste, previsoes)) #indica em porcentagem a quantidade de acertos

print(confusion_matrix(y_credito_teste, previsoes)) #retorna dois arrays, indicando: 
#[cliente que pagam e foram classificados como tal, cliente que não pagam e foram classificados como tal] 
#[clientes que não pagam e foram classificados como pagantes, clientes que não pagam e foram classificados como tal]

cm = ConfusionMatrix(naive_dados_credito)
cm.fit(x_credito_treinamento, y_credito_treinamento)
cm.score(x_credito_teste, y_credito_teste)
# print(cm.score(x_credito_teste, y_credito_teste)) também retorna a taxa de acerto

plt.show()  # o gráfico exibido exemplifica as classificações do seguinte modo:
# coordenada (0,0) clientes que pagam e foram classificados como tal
# coordenada (0,1) clientes que não pagam e foram classificados como pagantes
# coordenada (1,1) clientes que não pagam e foram classificados como tal
# coordenada (1,0) clientes que pagam e foram classificados como não pagantes

print(classification_report(y_credito_teste, previsoes)) # exibe os status de desempenho do algorítmo 

# recall significa o quanto ele consegue indificar corretamente dos registros 
# e após identificar, o precision indica a precisão de acertos