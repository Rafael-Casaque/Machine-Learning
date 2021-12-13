import pandas as pd
from scipy.stats import shapiro
import seaborn as sb
import matplotlib.pyplot as plt

resultados = pd.read_csv(r'Data-Science-and-Machine-Learning\Algorithm Evaluation\resultado-algoritmos.csv')

print(resultados.describe()) #imprime as estatísticas dos resultados dos algoritmos

#Com essas estatísticas é possível concluir que o algoritmo de redes neurais teve o melhor resultado, pois foi o melhor em accuracy e em desvio padrão (std)

print(resultados.var()) #imprime a variância dos resultados

print((resultados.std()/resultados.mean())*100) #imprime o coeficiente de variação (em percentual)

alpha = 0.05 #confiabilidade de 95% (padrão)

print(shapiro(resultados['arvores']))
print(shapiro(resultados['random forest']))
print(shapiro(resultados['Knn']))
print(shapiro(resultados['logistic regression']))
print(shapiro(resultados['svm']))
print(shapiro(resultados['redes neurais']))

#se o valor de p for menor ou igual a alpha, significa que os dados não são normais

#exibe um gráfico de distribuição

sb.displot(resultados['arvores'],kind='kde');
sb.displot(resultados['random forest'],kind='kde');
sb.displot(resultados['Knn'],kind='kde');
sb.displot(resultados['logistic regression'],kind='kde');
sb.displot(resultados['svm'],kind='kde');
sb.displot(resultados['redes neurais'],kind='kde');

plt.show()