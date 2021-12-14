import pickle 
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

with open('Data-Science-and-Machine-Learning\Algorithm Evaluation\credito.pkl', 'rb') as f:
    x_credito_treinamento,y_credito_treinamento,x_credito_teste, y_credito_teste = pickle.load(f)

x_credito = np.concatenate((x_credito_treinamento,x_credito_teste),axis=0) 

y_credito = np.concatenate((y_credito_treinamento,y_credito_teste),axis=0)

print(x_credito.shape,y_credito.shape)

classificador_neural_netork = MLPClassifier(activation='relu',batch_size=56,solver='adam')
classificador_neural_netork.fit(x_credito,y_credito)

classificador_arvores = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=1,min_samples_split=5,splitter='best')
classificador_arvores.fit(x_credito,y_credito)

classificador_svm = SVC(C=2.0,kernel='rbf')
classificador_svm.fit(x_credito,y_credito)

pickle.dump(classificador_neural_netork, open(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\final_neural_network.sav','wb'))
pickle.dump(classificador_arvores, open(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\final_arvores.sav','wb'))
pickle.dump(classificador_svm, open(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\final_svm.sav','wb'))

