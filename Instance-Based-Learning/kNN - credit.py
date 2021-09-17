import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Instance-Based-Learning\credito.pkl','rb') as f: # carrega o arquivo de dados
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f) # armazena os exercícios nas variáveis

print(x_credito_teste.shape,y_credito_teste.shape) #imprime a quantidade de registros teste

print(x_credito_treinamento.shape,y_credito_treinamento.shape) #imprime a quantidade de registros treinamento

knn_credito = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #n_neoghbors define o range, metric e p definem o cálculo da ditância

knn_credito.fit(x_credito_treinamento,y_credito_treinamento) # faz o "treinamento" que na verdade é apenas salvar os registros

previsoes = knn_credito.predict(x_credito_teste) # cria as previões das classes

print(accuracy_score(y_credito_teste,previsoes)) # informa o score de acerto

# gera a matriz de confusão 

cm = ConfusionMatrix(knn_credito)
cm.fit(x_credito_treinamento,y_credito_treinamento)
cm.score(x_credito_teste,y_credito_teste)

plt.show()

# Nesse gráfico:

# coordenada (0,0) pessoas que pagam e foram classificadas como tal
# coordenada (0,1) pessoas que não pagam e foram classificadas como pagantes
# coordenada (1,0) pessoas que pagam e foram classificadas como não pagantes
# coordenada (1,1) pessoas que não pagam e foram classificadas como tal

print(classification_report(y_credito_teste,previsoes))

# com relação á classe 0 o algorítmo consegue reconhecer 99% dos registros e tem precisão de 99%
# com relação á classe 1 o algorítmo consegue reconhecer 95% dos registros e tem precisão de 94%