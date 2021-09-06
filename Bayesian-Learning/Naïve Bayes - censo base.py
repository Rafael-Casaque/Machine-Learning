import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Bayesian-Learning\censo.pkl', 'rb') as f: #carrega o arquivo pré-processado anteriormente
    x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste = pickle.load(f) #armazena as informações do arquivo nas variáveis indicadas

print(x_censo_treinamento.shape, y_censo_treinamento.shape) # printa a quantidade de dados para treinamento 
print(x_censo_teste.shape, y_censo_teste.shape) # printa a quantidade de dados para teste 

naive_censo = GaussianNB()

naive_censo.fit(x_censo_treinamento,y_censo_treinamento) #dados para geração da tabela de probabilidade, primeiro parâmetro previsores, segundo a classe

previsoes = naive_censo.predict(x_censo_teste)

print(previsoes)

print(accuracy_score(y_censo_teste,previsoes)) # exibe o percentual de acerto

# Nesse caso, o resultado foi muito ruim, sendo mais preciso sortear no cara ou coroa do que usando esse algorítmo 

# Método para avaliação visual do desempenho do algorítmo
 
cm = ConfusionMatrix(naive_censo)
cm.fit(x_censo_treinamento, y_censo_treinamento)
cm.score(x_censo_teste,y_censo_teste)
plt.show()

# Esse gráfico exibe as informações da seguintte maniera: 

# coordenada (<=50k,<=50k) pessoas que ganham <=50k e foram classificadas como tal
# coordenada (>50k,<=50k) pessoas que ganham <=50k e foram classificadas como ganhando mais
# coordenada (>50k,>50k) pessoas que ganham <=50k e foram classificadas como tal
# coordenada (<=50k,>50k) pessoas que ganham >50k e foram classificadas como ganhando menos

print(classification_report(y_censo_teste,previsoes))

# resultado: 

# o algorítmo consegue identificar apenas 32% das pessoas que ganham <=50k, porém, quando identificado, a precisão é de 97%
# o algorítmo consegue identificar apenas 97% das pessoas que ganham >50k, porém, quando identificado, a precisão é de 31%