import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier

with open('Data-Science-and-Machine-Learning\Algorithm Evaluation\credito.pkl', 'rb') as f:
    x_credito_treinamento,y_credito_treinamento,x_credito_teste, y_credito_teste = pickle.load(f)

#print(x_credito_treinamento.shape,y_credito_treinamento.shape)
#print(x_credito_teste.shape,y_credito_teste.shape)

#faz a concatenação das base de dados para a validação cruzada.

x_credito = np.concatenate((x_credito_treinamento,x_credito_teste),axis=0) 
#print(x_credito.shape)

y_credito = np.concatenate((y_credito_treinamento,y_credito_teste),axis=0)
#print(y_credito.shape)

#Tree Decision

print("\nTree Decision\n")

#cria um dicionários com os parâmetros randomicos para serem testados

parametros = {'criterion':['gini','entropy'],'splitter':['best','random'],'min_samples_split':[2,5,10],'min_samples_leaf':[1,5,10]}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=parametros)

grid_search.fit(x_credito,y_credito)

melhores_parametros = grid_search.best_params_

melhor_resultado = grid_search.best_score_

print("melhores parâmetros: ",melhores_parametros)
print("melhor resultado: ",melhor_resultado)



#Random Forest

print("\nRandom Forest\n")


parametros = {'criterion':['gini','entropy'],'n_estimators':[10,40,100,150],'min_samples_split':[2,5,10],'min_samples_leaf':[1,5,10]}

grid_search = GridSearchCV(estimator=RandomForestClassifier(),param_grid=parametros)

grid_search.fit(x_credito,y_credito)

melhores_parametros = grid_search.best_params_

melhor_resultado = grid_search.best_score_

print("melhores parâmetros: ",melhores_parametros)
print("melhor resultado: ",melhor_resultado)



#Knn

print("\nKnn\n")


parametros = {'n_neighbors':[3,5,10,20],'p':[1,2]}

grid_search = GridSearchCV(estimator=KNeighborsClassifier(),param_grid=parametros)

grid_search.fit(x_credito,y_credito)

melhores_parametros = grid_search.best_params_

melhor_resultado = grid_search.best_score_

print("melhores parâmetros: ",melhores_parametros)
print("melhor resultado: ",melhor_resultado)

#Logistic Regression

print("\nLogistic Regression\n")


parametros = {'tol':[0.001,0.00001,0.000001],'C':[1.0,1.5,2.0],'solver':['lbfgs','sag','saga']}

grid_search = GridSearchCV(estimator=LogisticRegression(),param_grid=parametros)

grid_search.fit(x_credito,y_credito)

melhores_parametros = grid_search.best_params_

melhor_resultado = grid_search.best_score_

print("melhores parâmetros: ",melhores_parametros)
print("melhor resultado: ",melhor_resultado)




#SVM

print("\nSVM\n")


parametros = {'tol':[0.001,0.0001,0.00001],'C':[1.0,1.5,2.0],'kernel':['rbf','linear','poly','sigmoid']}

grid_search = GridSearchCV(estimator=SVC(),param_grid=parametros)

grid_search.fit(x_credito,y_credito)

melhores_parametros = grid_search.best_params_

melhor_resultado = grid_search.best_score_

print("melhores parâmetros: ",melhores_parametros)
print("melhor resultado: ",melhor_resultado)




#Artificial Neural Network

print("\nArtificial Neural Network\n")


parametros = {'activation':['relu','logistic','tahn'],'solver':['adam','sgd'],'batch_size':[10,56]}

grid_search = GridSearchCV(estimator=MLPClassifier(),param_grid=parametros)

grid_search.fit(x_credito,y_credito)

melhores_parametros = grid_search.best_params_

melhor_resultado = grid_search.best_score_

print("melhores parâmetros: ",melhores_parametros)
print("melhor resultado: ",melhor_resultado)