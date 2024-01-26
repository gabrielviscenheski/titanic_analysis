#TITANIC - Análise Inicial

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

#**Carregando os Dados:**

dataset_titanic_test = pd.read_csv("test.csv")
dataset_titanic_train = pd.read_csv("train.csv")

#**Análise Exploratória:**

# Primeiras linhas
dataset_titanic_train.head()

# Últimas linhas
dataset_titanic_train.tail()

# Info
dataset_titanic_train.info()

# Entendendo os valores NaN
(dataset_titanic_train.isnull().sum()/dataset_titanic_train.shape[0]).sort_values(ascending=True)

# Resumo Estatístico
dataset_titanic_train.describe()

#**Análise Inicial:**
# 1) Colunas com alta cardinalidade podem ser excluídas: em uma primeira análise dificulta a predição do modelo
# 2) Colunas numéricas: alvo dessa primeira análise
# 3) Valores nulos: Cabin, Age, Embarked

#**Verificação: Distribuição Normal**

# 1) Coluna Age:**
# Possível substituição dos valores nulos pela média de idade e para isso é necessário verificar se há uma distribuição normal dos dados

#**Histograma:**

# Verificando se há uma distribuição normal com a coluna de idade
plt.hist(dataset_titanic_train["Age"], histtype = "stepfilled", rwidth = 0.8)
plt.xlabel("Idade")
plt.ylabel("Frequência")

#**KDEPlot:**

# Verificando se há uma distribuição normal com a coluna de idade
sea.kdeplot(data=dataset_titanic_train["Age"], shade=True)

#**Tratamento de Dados (train.csv):**

#**1) Eliminação das colunas "Name", "Ticket" e "Cabin":**

# Eliminação das colunas abaixo pela alta cardinalidade
dataset_titanic_train = dataset_titanic_train.drop(["Name","Ticket","Cabin"],axis=1)

# Verificação da exclusão das colunas
dataset_titanic_train.columns

#**2) Substituição de valores nulos pela média de idade na coluna "Age"**

# Há uma distribuição normal dos dados e pouca presença de outliers, em razão disso, é viável aplicar a média.**

# Média de idade
filter1_mean = dataset_titanic_train["Age"].mean()

# Substituindo na coluna "Age"
dataset_titanic_train.loc[dataset_titanic_train.Age.isnull(),"Age"] = filter1_mean

# Verificação da substituição
dataset_titanic_train.info()

#**3) Substituição dos valores nulos no "Embarked" pela moda:**

# A distribuição é não normal e, há poucos valores únicos, em razão disso é viavel aplicar a moda.**

# Moda na coluna Embarked
filter2_mode = dataset_titanic_train["Embarked"].mode()[0]
filter2_mode

# Substituição dos NaN pela moda
dataset_titanic_train.loc[dataset_titanic_train.Embarked.isnull(),"Embarked"] = filter2_mode

# Verificação da substituição
dataset_titanic_train.info()

#**Curiosidade:**

#**KNNImputer é outra abordagem para valores nulos preenchendo os valores nulos através de vizinhos próximos**

from sklearn.impute import KNNImputer

# X = dataset_titanic_train.drop(columns=['Age', 'Sex', 'Embarked'])
# imputer = KNNImputer(n_neighbors=15)
# predict = imputer.fit_transform(X)
# predict

#**Tratamento de Dados (test.csv):**

#**1) Eliminação das colunas "Name", "Ticket" e "Cabin":**
"""

# Eliminação das colunas abaixo pela alta cardinalidade
dataset_titanic_test = dataset_titanic_test.drop(["Name","Ticket","Cabin"],axis=1)

# Verificação da exclusão das colunas
dataset_titanic_test.columns

#**2) Substituição de valores nulos pela média de idade na coluna "Age"**

# Média de idade
filter1_mean = dataset_titanic_test["Age"].mean()
filter1_mean

# Substituindo na coluna "Age"
dataset_titanic_test.loc[dataset_titanic_test.Age.isnull(),"Age"] = filter1_mean

# Verificação da substituição
dataset_titanic_test.info()

#**Substituição dos valores nulos no "Embarked" pela moda:**

# Não precisa pois apresenta 0 valores nulos

#**Exclusão do valor nulo na coluna Fare com a mediana:**

# Verificar se há uma distribuição normal
sea.kdeplot(data=dataset_titanic_test["Fare"], shade=True)

# Alto número de outliers
sea.boxplot(dataset_titanic_test["Fare"])

# Criação do filtro
filter1_median = dataset_titanic_test["Fare"].median()
filter1_median

# Substituição dos valores nulos pela mediana
dataset_titanic_test.loc[dataset_titanic_test.Fare.isnull(),"Fare"] = filter1_median

# Verificação da substituição
dataset_titanic_test.isnull().sum()

#**Preparação dos Dados para o Modelo:**

# Colunas de valores numéricos da base de treino
colunas_train_nr = dataset_titanic_train.columns[dataset_titanic_train.dtypes != "object"]

# Somente linhas e colunas de valores numéricos
dataset_titanic_train_num = dataset_titanic_train.loc[:,colunas_train_nr]

# Colunas de valores numéricos da base de teste
colunas_test_nr = dataset_titanic_test.columns[dataset_titanic_test.dtypes != "object"]

# Somente linhas e colunas de valores numéricos
dataset_titanic_test_num = dataset_titanic_test.loc[:,colunas_test_nr]

# Importando o train_test_split
from sklearn.model_selection import train_test_split

# Input variables
X = dataset_titanic_train_num.drop(["PassengerId","Survived"],axis=1)

# Target variable
y = dataset_titanic_train.Survived

# Treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20 , random_state = 42)

#**Modelos de Classificação:**

**Regressão Logística:**

# Importação
from sklearn.linear_model import LogisticRegression

# Criação do classificador
clf_rl = LogisticRegression(random_state=0).fit(X, y)

# Data Fit
clf_rl = clf_rl.fit(X_train,y_train)

# Previsão
y_pred_rl = clf_rl.predict(X_val)

#**Árvore de Classificação:**

# Importação
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Criação do classificador
clf_tree = DecisionTreeClassifier(random_state=0)

y_train = y_train.astype('category')

# Data Fit
clf_tree = clf_tree.fit(X_train, y_train)

# Previsão
y_pred_tree = clf_tree.predict(X_val)

#**Avaliação do Modelo:**

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Accuracy: Regressão Logística
accuracy_score(y_val,y_pred_rl)

# Accuracy: Árvore de Decisão
accuracy_score(y_val,y_pred_tree)

# Matriz de Confusão : Regressão Logística
confusion_matrix(y_val, y_pred_rl)

# Matriz de Confusão: Árvore de Decisão
confusion_matrix(y_val, y_pred_tree)

#**Previsão para a Base de Teste:**

# Base de teste ser igual a base de treino: eliminar PassengerId
X_teste = dataset_titanic_test_num.drop("PassengerId", axis =1)

# Regressão Logística na base de teste
y_pred = clf_rl.predict(X_teste)

# Nova coluna com a previsão
dataset_titanic_test_num["Survived"] = y_pred

# PassengerId e Survived
dataset_final = dataset_titanic_test_num[["PassengerId","Survived"]]
dataset_final.shape

# Envio
dataset_final.to_csv("dataset_final.csv",index=False)
dataset_final
