TPC 3
Crie um ficheiro em python para trabalhar o dataset
    datasets.california_housing
Nesse ficheiro, crie um script (função) por alínea que lhe permita gerar novos datasets a partir do dataset principal, onde tenha usado cada um dos seguintes métodos de pre-processamento:
1) Aggregation
2) Sampling
4) Dimensionality Reduction 
5) Feature Subset Selection 
6) Feature Creation 
7) Discretization and Binarization 
8) Attribute Transformation
O que é feito em cada caso, é da sua inteira liberdade.
[41]

# Importação dos Módulos
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
#Importação das variavéis
data = datasets.california_housing.fetch_california_housing()
​
#show disorganized data
data
[ ]

# 1- Agregação do máximo, mínimo, média da Feature Poputação
​
dframe = pd.DataFrame(data = data.data, columns=data.feature_names)
dframe.head()
​
dframe.aggregate(['max', 'min', 'mean'])['Population']
[49]

# 2- Sampling
​
def sample_stat(sample):
  print(sample)
  return sample.mean(['Population'])
[37]

# 3- Redução da dimencionalidade
dframe.loc[0:1,["MedInc"]]
  
[ ]

# 4- Feature Subset Selection
​
​
​
[35]

# 5-  Feature Creation
# Foi criada uma Feature para agrupar o tamanho da população e o idade da casa
​
dframe['FamilySize'] = dframe['Population'] + dframe['HouseAge']
dframe.head()
[38]

# 6- Discretization and Binarization
​
from sklearn import preprocessing
​
binarizer = preprocessing.Binarizer().fit('Population')
print(binarizer)
binarizer.threshold=3.50
[50]

# 7- Attribute Transformation
​
def draw_missing_data_table(dframe):
    total = dframe.isnull().sum().sort_values(ascending=False)
    percent = (dframe.isnull().sum()/dframe.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
  
​
​
​
