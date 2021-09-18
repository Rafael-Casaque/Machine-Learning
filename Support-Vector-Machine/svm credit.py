from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Support-Vector-Machine\credito.pkl','rb') as f:
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f)

print(x_credito_teste.shape,y_credito_teste.shape) # printa a quantidade de registros teste

print(x_credito_treinamento.shape,y_credito_treinamento.shape) # printa a quantidade de registros treinamento

svm_credito = SVC(kernel='rbf',random_state=1, C = 2) # cria o algorítmo pasando os parâmetros kernel aceita: linear,poly,sigmoid,rbf. Quanto maior o parâmetro C, maior as chances de encontrar os melhores vetore

svm_credito.fit(x_credito_treinamento,y_credito_treinamento) # faz a aprendizagem para encontrar os vetores

previsoes = svm_credito.predict(x_credito_teste) # faz as previsões das classes

print(accuracy_score(y_credito_teste,previsoes))

cm = ConfusionMatrix(svm_credito)
cm.fit(x_credito_treinamento,y_credito_treinamento)
cm.score(x_credito_teste,y_credito_teste)

plt.show()

# Nesse gráfico:

# coordenada (0,0) pessoas que pagam e foram classificadas como tal
# coordenada (0,1) pessoas que não pagam e foram classificadas como pagantes
# coordenada (1,0) pessoas que pagam e foram classificadas como não pagantes
# coordenada (1,1) pessoas que não pagam e foram classificadas como tal

print(classification_report(y_credito_teste,previsoes))

# com relação á classe 0 o algorítmo consegue reconhecer 100% dos registros e tem precisão de 99%
# com relação á classe 1 o algorítmo consegue reconhecer 97% dos registros e tem precisão de 94%
