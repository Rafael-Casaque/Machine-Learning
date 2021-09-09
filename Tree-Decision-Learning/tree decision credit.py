from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('Data-Science-and-Machine-Learning\Tree-Decision-Learning\credito.pkl','rb') as f:
    x_credito_treinamento, y_credito_treinamento, x_credito_teste, y_credito_teste = pickle.load(f)

print(x_credito_treinamento.shape,y_credito_treinamento.shape) # exibe o tamanho dos registros a serem usados para treinar o algorítmo

print(x_credito_teste.shape,y_credito_teste.shape) # exibe o tamanho dos registros a serem usados para testar o algorítmo

arvore_credito = DecisionTreeClassifier(criterion='entropy', random_state= 0) # cria uma variável para o aprendizado, tendo como critério de importância, a entropia

arvore_credito.fit(x_credito_treinamento,y_credito_treinamento) # faz o aprendizado a partir dos registros de treinamento 

previsoes = arvore_credito.predict(x_credito_teste) # faz as previsões a partir dos regitros teste

print(previsoes) # exibe as previsões

print(accuracy_score(y_credito_teste,previsoes)) # exibe a porcentagem de acerto, comparando os registros reais com as previsões

cm = ConfusionMatrix(arvore_credito)

cm.fit(x_credito_treinamento,y_credito_treinamento)

cm.score(x_credito_teste, y_credito_teste)

plt.show()

# Esse gráfico exibe as informações da seguintte maniera: 

# coordenada (0,0) pessoas que pagam e foram classificadas como tal
# coordenada (0,1) pessoas que não pagam e foram classificadas como pagantes
# coordenada (1,0) pessoas que pagam e foram classificadas como não pagantes
# coordenada (1,1) pessoas que não pagam e foram classificadas como tal

print(classification_report(y_credito_teste,previsoes)) 

# com relação á classe 0 o algorítmo consegue reconhecer 99% dos registros e tem precisão de 99%
# com relação á classe 1 o algorítmo consegue reconhecer 95% dos registros e tem precisão de 91%

print(arvore_credito.feature_importances_) # exibe o nível de importância de cada atributo (ganho de informação)

print(tree.plot_tree(arvore_credito)) #exibe a árvore de decisão em formato de texto

plt.show() #exibe a árvore de decisão em formato de fluxograma

previsores = ['renda', 'idade', 'dívida']
tree.plot_tree(arvore_credito, feature_names=previsores)

plt.show() #exibe a árvore de decisão em formato de fluxograma com os nomes presentes na variável previsores

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (45,35)) #cria a representação da árvore em uma fig com 1x1 de tamanho 20x20
tree.plot_tree(arvore_credito, feature_names=previsores, class_names=['paga','não paga'], filled=True); #faz a definição dos nomes, inclive de classes
fig.savefig(r'Data-Science-and-Machine-Learning\Tree-Decision-Learning\arvore_credito.png',bbox_inches='tight') #salva a imagem (usa-se o r para ignorar a \)


