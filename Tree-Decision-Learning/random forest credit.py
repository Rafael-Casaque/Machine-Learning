from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle


with open('Data-Science-and-Machine-Learning\Tree-Decision-Learning\credito.pkl','rb') as f: # carrega o arquivo na variável f
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f) # carrega as variáveis com o arquivo da variável f

print(x_credito_treinamento.shape,y_credito_treinamento.shape) # informa a quantidade de registros de treinamento

print(x_credito_teste.shape,y_credito_teste.shape) # informa a quantidade de registros de teste

random_forest_credito = RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=0) # cria a floresta com 40 árvores de decisão e entropia como criterio de relevância

random_forest_credito.fit(x_credito_treinamento,y_credito_treinamento) # faz o treinamento do algoritmo com os registros de treinamento

previsoes = random_forest_credito.predict(x_credito_teste) # cria as pevisoes para os registros teste

print(accuracy_score(y_credito_teste,previsoes)) # informa o percentual de acerto

cm = ConfusionMatrix(random_forest_credito)

cm.fit(x_credito_treinamento,y_credito_treinamento)

cm.score(x_credito_teste,y_credito_teste)

plt.show()

# Nesse gráfico

#(<=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como tal
#(>=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como ganhando mais
#(<=50k,>=50k) são pessoas que não ganham menos de 50k e foram classificados como tal
#(>=50k,>=50k) são pessoas que ganham mais de 50k e foram classificados como tal