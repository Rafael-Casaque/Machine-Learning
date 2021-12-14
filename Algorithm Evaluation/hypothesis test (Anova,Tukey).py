from scipy.stats import f_oneway
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt

resultado = pd.read_csv(r"Data-Science-and-Machine-Learning\Algorithm Evaluation\resultado-algoritmos.csv")

#O '_,' ignora o primeiro resultado obtido

_, p = f_oneway(resultado['arvores'],resultado['random forest'],resultado['Knn'],resultado['logistic regression'],resultado['svm'],resultado['redes neurais'])

alpha = 0.05

print(p)

#Se o valor de p for <= que alpha, significa que os dados obtidos são diferentes, caso contrário, ignifica que são iguais (não existem diferenças estatísticas)

#Se os resultados fossem iguais, não faria diferença o algoritmo utilizado

resultados_algoritmos = {'accuracy':np.concatenate([resultado['arvores'],resultado['random forest'],resultado['Knn'],resultado['logistic regression'],resultado['svm'],resultado['redes neurais']])}

resultados_algoritmos = pd.DataFrame(resultados_algoritmos)

resultados_algoritmos = resultados_algoritmos.assign(algoritmo=None)
resultados_algoritmos['algoritmo'][0:30]='arvores'
resultados_algoritmos['algoritmo'][30:60]='random forest'
resultados_algoritmos['algoritmo'][60:90]='Knn'
resultados_algoritmos['algoritmo'][90:120]='logistic regression'
resultados_algoritmos['algoritmo'][120:150]='svm'
resultados_algoritmos['algoritmo'][150:180]='redes neurais'

comparacao = MultiComparison(resultados_algoritmos['accuracy'],resultados_algoritmos['algoritmo'])

teste_estatistico = comparacao.tukeyhsd()

print(teste_estatistico)

print(resultados_algoritmos.mean())

teste_estatistico.plot_simultaneous();
plt.show()

#Dessa forma é possível concluir que a rede neural tem o melhor resultado de todos