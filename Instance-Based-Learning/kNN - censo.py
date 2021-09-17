import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Instance-Based-Learning\censo.pkl','rb') as f: # carrega o arquivo de dados
    x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste = pickle.load(f) # armazena os exercícios nas variáveis

print(x_censo_teste.shape,y_censo_teste.shape) #imprime a quantidade de registros teste

print(x_censo_treinamento.shape,y_censo_treinamento.shape) #imprime a quantidade de registros treinamento

knn_censo = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2) #n_neighbors define o range, metric e p definem o cálculo da ditância

knn_censo.fit(x_censo_treinamento,y_censo_treinamento) # faz o "treinamento" que na verdade é apenas salvar os registros

previsoes = knn_censo.predict(x_censo_teste) # cria as previões das classes

print(accuracy_score(y_censo_teste,previsoes)) # informa o score de acerto

# gera a matriz de confusão 

cm = ConfusionMatrix(knn_censo)
cm.fit(x_censo_treinamento,y_censo_treinamento)
cm.score(x_censo_teste,y_censo_teste)

plt.show()

# Nesse gráfico

#(<=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como tal
#(>=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como ganhando mais
#(<=50k,>=50k) são pessoas que não ganham menos de 50k e foram classificados como tal
#(>=50k,>=50k) são pessoas que ganham mais de 50k e foram classificados como tal

print(classification_report(y_censo_teste,previsoes))

# para os que ganham <=50k ele reconhece em 93% dos casos e tem precisão de 86%
# para os que ganham >50k ele reconhece em 51% dos casos e tem precisão de 71%