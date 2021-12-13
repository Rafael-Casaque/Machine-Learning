from sklearn.model_selection import cross_val_score, KFold

import pandas as pd
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

x_credito = np.concatenate((x_credito_treinamento,x_credito_teste),axis=0) 

y_credito = np.concatenate((y_credito_treinamento,y_credito_teste),axis=0)

resultado_arvore = []
resultado_random_forest = []
resultado_knn = []
resultado_logistic_regression = []
resultado_svm = []
resultado_neural_network = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True,random_state=i) 
    
    #n_plits é o número de divisão feita na base de dados, utilizando a cada rodada 9 pra treinamento e 1 pra teste 
    #shuffle true faz com que os resultados sempre sejam misturados
    #random_state significa que os resultados seram sempre diferentes

    #cria o algoritmo de arvores de decisão com os melhores parâmetros, obtidos no tunning
    
    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1,min_samples_split=5,splitter='best') 
    
    #realiza o teste e exibe o accuracy dos modos como foram testados 

    scores = cross_val_score(arvore,x_credito,y_credito,cv = kfold)

    resultado_arvore.append(scores.mean()) # adiciona a media de cada rodada de testes 

    random_forest = RandomForestClassifier(criterion='entropy',min_samples_leaf=1,min_samples_split=5,n_estimators=10)
    scores = cross_val_score(random_forest,x_credito,y_credito,cv = kfold)
    resultado_random_forest.append(scores.mean()) # adiciona a media de cada rodada de testes 

    knn = KNeighborsClassifier()
    scores = cross_val_score(knn,x_credito,y_credito,cv = kfold)
    resultado_knn.append(scores.mean())

    logistic_regression = LogisticRegression(C=1.0,solver='lbfgs',tol=0.0001)
    scores = cross_val_score(logistic_regression,x_credito,y_credito,cv = kfold)
    resultado_logistic_regression.append(scores.mean())

    svm = SVC(kernel='rbf',C=2.0)
    scores = cross_val_score(svm,x_credito,y_credito,cv = kfold)
    resultado_svm.append(scores.mean())

    neural_networks = MLPClassifier(activation='relu',batch_size=56,solver='adam')
    scores = cross_val_score(neural_networks,x_credito,y_credito,cv = kfold)
    resultado_neural_network.append(scores.mean())        

#criando um dataframe com os resultados obtidos e escrevendo em um arquivo csv

resultado = pd.DataFrame({'arvores':resultado_arvore,'random forest':resultado_random_forest,'Knn':resultado_knn,'logistic regression':resultado_logistic_regression,'svm':resultado_svm,'redes neurais':resultado_neural_network})
resultado.to_csv('Data-Science-and-Machine-Learning\Algorithm Evaluation\resultado-algoritmos.csv',index=False)