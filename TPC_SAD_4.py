Aplicar Algoritmos
Usando sklearn correr os métodos Decision Tree, Random Forrest e Naive Bayes para o dataset Digits, definido em baixo;
Usar o training set para executar o treino do modelo;
Comparar o erro obtido em cada método, para o testset e para o training set e expecificar se os valores são os esperados;
Para um dos algoritmos, dar exemplos do test set de instâncias mal bem classificadas (2 de cada);
O DataSet de dígitos pode ser carregado assim:
[2]

​
# Import datasets, classifiers and performance metrics
from sklearn import datasets, tree, model_selection
​
# The digits dataset
digits = datasets.load_digits()
​
# Mostrar o dataset
​
digits
​
# Definição dos dados para treino
[features_train, features_test, classes_train, classes_test] = model_selection.train_test_split(digits.data, digits.target, test_size=0.30)
model = tree.DecisionTreeClassifier()
​
clf = model.fit(features_train, classes_train)
​
score_train = model.score(features_train, classes_train)
score_test = model.score(features_test, classes_test)
​
print("Features:", digits.target_names)
print("score_train:", score_train)
print("score_test:", score_test)
​
​
​
# Explicação do resultado obitido
# Podemos notar que od dados para o treino foram bem classificados, dando
# então o valor de 100%, mas os dados para o teste dá um valor não aprocimado a 100%
​
​
​
​

[3]

# Usando Random Forrest com sklearn
​
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
​
digits = datasets.load_digits()
​
​
# Definição dos dados para treino
[features_train, features_test, classes_train, classes_test] = model_selection.train_test_split(digits.data, digits.target, test_size=0.30)
model = RandomForestClassifier(n_estimators=1000)
​
clf = model.fit(features_train, classes_train)
​
score_train = model.score(features_train, classes_train)
score_test = model.score(features_test, classes_test)
​
print("Features:", digits.target_names)
print("score_train:", score_train)
print("score_test:", score_test)
​
​
# Explicação do resultado obitido
# Podemos notar que neste método, obtemos  que os dados para o treino dá 100%
# Sendo os dados para o teste dá um valor aproximado a 100%

[4]

​
# Usando o método Naive Bayes
​
from sklearn import datasets
digits = datasets.load_digits()
from sklearn import datasets, tree, model_selection
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
​
# Definição dos dados para treino
[features_train, features_test, classes_train, classes_test] = model_selection.train_test_split(digits.data, digits.target, test_size=0.30)
y_pred = gnb.fit(features_train, classes_train)
​
score_train = gnb.score(features_train, classes_train)
score_test = gnb.score(features_test, classes_test)
​
print("Features:", digits.target_names)
print("score_train:", score_train)
print("score_test:", score_test)
​
​
​
# Explicação do resultado obitido
# Neste metódo, tatnto as valores para o treinoe para o teste são aproximadamente semelhantes.
# Tanto os dados de treino como os dados de teste não dão aproximadamente 100% estando um pouco distante

Conclusão, para os o algoritmo de Naive Bayes, podemo notar que os dados usado para o test, são o resultado do score_test dá mais baixo em relação ao score de treino, podemos notar que pelo facto de considerar as features independentes umas das outras, classificou mal os digitos, não distiguindo claramente os valores como 1 ou 0.