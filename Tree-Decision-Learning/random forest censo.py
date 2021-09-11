from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle


with open('Data-Science-and-Machine-Learning\Tree-Decision-Learning\censo.pkl','rb') as f: # carrega o arquivo na variável f
    x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste = pickle.load(f) # carrega as variáveis com o arquivo da variável f

print(x_censo_treinamento.shape,y_censo_treinamento.shape) # informa a quantidade de registros de treinamento

print(x_censo_teste.shape,y_censo_teste.shape) # informa a quantidade de registros de teste

random_forest_credito = RandomForestClassifier(n_estimators=120,criterion='entropy',random_state=0) # cria a floresta com 120 árvores de decisão e entropia como criterio de relevância

random_forest_credito.fit(x_censo_treinamento,y_censo_treinamento) # faz o treinamento do algoritmo com os registros de treinamento

previsoes = random_forest_credito.predict(x_censo_teste) # cria as pevisoes para os registros teste

print(accuracy_score(y_censo_teste,previsoes)) # informa o percentual de acerto

cm = ConfusionMatrix(random_forest_credito)

cm.fit(x_censo_treinamento,y_censo_treinamento)

cm.score(x_censo_teste,y_censo_teste)

plt.show()

# Nesse gráfico

#(<=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como tal
#(>=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como ganhando mais
#(<=50k,>=50k) são pessoas que não ganham menos de 50k e foram classificados como tal
#(>=50k,>=50k) são pessoas que ganham mais de 50k e foram classificados como tal

print(classification_report(y_censo_teste,previsoes))

# para os que ganham <=50k ele reconhece em 93% dos casos e tem precisão de 88%
# para os que ganham >50k ele reconhece em 62% dos casos e tem precisão de 73%