from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt


with open('Data-Science-and-Machine-Learning\Artificial-Neural-Networks\credito.pkl','rb') as f:
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f)

rede_neural_credito = MLPClassifier(max_iter=1500,verbose=True,tol=0.00001,solver='adam',activation='relu',hidden_layer_sizes=(2,2))

# parâmetros utilizados:

# max_iter define o valor de iterações máximas que a rede neural deverá ter
# verbose True faz com que as iterações sejam printadas
# tol se refere à diferença de melhoria de erro de uma época para outra
# solver se refere ao algorítmo utilizado para redefinir os pesos
# activation se refere à função de ativação utilizada
# hidden_layer_sizes se refere à quantidade de neurônios na camada de saída

rede_neural_credito.fit(x_credito_treinamento, y_credito_treinamento)

previsoes = rede_neural_credito.predict(x_credito_teste)
print(accuracy_score(y_credito_teste,previsoes))

cm = ConfusionMatrix(rede_neural_credito)
cm.fit(x_credito_treinamento, y_credito_treinamento)
cm.score(x_credito_teste,y_credito_teste)

plt.show()