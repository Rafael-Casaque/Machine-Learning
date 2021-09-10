from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open(r'Data-Science-and-Machine-Learning\Tree-Decision-Learning\censo.pkl', 'rb') as f: #abre o arquivo na variável f
    x_censo_treinamento, y_censo_treinamento, x_censo_teste, y_censo_teste = pickle.load(f) # atribui os regitros em cada variável

print(x_censo_treinamento.shape,y_censo_treinamento.shape)    # printa a quantidade de regitros para treinamento 

print(x_censo_teste.shape,y_censo_teste.shape) # printa a quantidade de regitros para teste

arvore_censo = DecisionTreeClassifier(criterion='entropy',random_state=0) # faz a criação da árvore

arvore_censo.fit(x_censo_treinamento,y_censo_treinamento) # faz o aprendizado do algorítmo

previsoes = arvore_censo.predict(x_censo_teste) # faz as previsões

print(accuracy_score(y_censo_teste,previsoes)) # imprime o percentual de acerto

cm = ConfusionMatrix(arvore_censo)

cm.fit(x_censo_treinamento,y_censo_treinamento)

cm.score(x_censo_teste, y_censo_teste)

plt.show()

# Nesse gráfico

#(<=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como tal
#(>=50k,<=50k) são pessoas que ganham menos de 50k e foram classificados como ganhando mais
#(<=50k,>=50k) são pessoas que não ganham menos de 50k e foram classificados como tal
#(>=50k,>=50k) são pessoas que ganham mais de 50k e foram classificados como tal

print(classification_report(y_censo_teste,previsoes))

# para os que ganham <=50k ele reconhece em 87% dos casos e tem precisão de 88%
# para os que ganham >50k ele reconhece em 61% dos casos e tem precisão de 61%