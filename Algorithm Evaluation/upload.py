import pickle
import numpy as np

with open('Data-Science-and-Machine-Learning\Algorithm Evaluation\credito.pkl', 'rb') as f:
    x_credito_treinamento,y_credito_treinamento,x_credito_teste, y_credito_teste = pickle.load(f)

x_credito = np.concatenate((x_credito_treinamento,x_credito_teste),axis=0) 

y_credito = np.concatenate((y_credito_treinamento,y_credito_teste),axis=0)

rede_neural = pickle.load(open(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\final_neural_network.sav','rb'))
arvore = pickle.load(open(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\final_arvores.sav','rb'))
svm = pickle.load(open(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\final_svm.sav','rb'))

novo_registro = x_credito[0]

print(novo_registro)

novo_registro = novo_registro.reshape(1,-1)

#realiza a predição de dados

print(rede_neural.predict(novo_registro))
print(arvore.predict(novo_registro))
print(svm.predict(novo_registro))