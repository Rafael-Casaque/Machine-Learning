import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Logistic-Regression\censo.pkl','rb') as f:
    x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste = pickle.load(f)

print(x_censo_teste.shape,y_censo_teste.shape) # printa a quantidade de registros teste

print(x_censo_treinamento.shape,y_censo_treinamento.shape) # printa a quantidade de registros treinamento

# Faz a criação do modelo de aprendizagem

logist_censo = LogisticRegression(random_state=1)
logist_censo.fit(x_censo_treinamento,y_censo_treinamento)

print(logist_censo.intercept_) # informa o valor de b0

print(logist_censo.coef_) # informa o coeficiente de cada atributo

previsoes = logist_censo.predict(x_censo_teste) # cria as previsões para esse algoritmo

print(accuracy_score(y_censo_teste,previsoes)) # exibe a porcentagem de acerto

cm = ConfusionMatrix(logist_censo)
cm.fit(x_censo_treinamento,y_censo_treinamento)
cm.score(x_censo_teste,y_censo_teste)

plt.show()

# Nesse gráfico:

#(<=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como tal
#(>=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como ganhando mais
#(<=50k,>=50k) são pessoas que não ganham menos de 50k e foram classificados como tal
#(>=50k,>=50k) são pessoas que ganham mais de 50k e foram classificados como tal

print(classification_report(y_censo_teste,previsoes))

# para os que ganham <=50k ele reconhece em 93% dos casos e tem precisão de 88%
# para os que ganham >50k ele reconhece em 61% dos casos e tem precisão de 73%