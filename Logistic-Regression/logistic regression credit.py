import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Logistic-Regression\credito.pkl','rb') as f:
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f)

print(x_credito_teste.shape,y_credito_teste.shape) # printa a quantidade de registros teste

print(x_credito_treinamento.shape,y_credito_treinamento.shape) # printa a quantidade de registros treinamento

# Faz a criação do modelo de aprendizagem

logist_credito = LogisticRegression(random_state=1)
logist_credito.fit(x_credito_treinamento,y_credito_treinamento)

print(logist_credito.intercept_) # informa o valor de b0

print(logist_credito.coef_) # informa o coeficiente de cada atributo

previsoes = logist_credito.predict(x_credito_teste) # cria as previsões para esse algoritmo

print(accuracy_score(y_credito_teste,previsoes)) # exibe a porcentagem de acerto

cm = ConfusionMatrix(logist_credito)
cm.fit(x_credito_treinamento,y_credito_treinamento)
cm.score(x_credito_teste,y_credito_teste)

plt.show()

# Nesse gráfico:

# coordenada (0,0) pessoas que pagam e foram classificadas como tal
# coordenada (0,1) pessoas que não pagam e foram classificadas como pagantes
# coordenada (1,0) pessoas que pagam e foram classificadas como não pagantes
# coordenada (1,1) pessoas que não pagam e foram classificadas como tal

print(classification_report(y_credito_teste,previsoes))

# com relação á classe 0 o algorítmo consegue reconhecer 97% dos registros e tem precisão de 97%
# com relação á classe 1 o algorítmo consegue reconhecer 78% dos registros e tem precisão de 79%