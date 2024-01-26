#TITANIC - Análise Intermediária 1

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

#**Insights 1:**

# * Todas as colunas com tipos de dados coerentes.
# * Resumo Estatístico:
# * Mean: PassengerID (não importa) ;
# * Survived (menor que 50%) ;
# * PClass (tendendo pra 3a classe) ;
# * Age (próximo dos 30) ;
# * Sibsp(com 0 ou 1 conjugê) ;
# * Parch (próximo de 0 acompanhantes) ;
# * Colunas com valores nulos: Cabin, Age, Embarked
# * Simplesmente excluir as linhas com valores nulos pode causar uma diminuição drástica do dataset, afetando no modelo.
# * Colunas com alta cardinalidade podem ser excluídas para análise de um primeiro modelo:
#   * Name: já temos outras identificação, PassengerId
#   * Ticket: pré análise guia para uma não correlação com a variável alvo.
#   * Cabin: será abordada em uma análise mais refinada.

#**Tratamento de Dados (train.csv): Parte I**

#**1) Eliminação das colunas "Name", "Ticket" e "Cabin":**

# Justificativa:** Alta cardinalidade ou redundância

# Eliminação das colunas abaixo pela alta cardinalidade
dataset_titanic_train = dataset_titanic_train.drop(["Name","Ticket","Cabin"],axis=1)

# Verificação da exclusão das colunas
dataset_titanic_train.columns

#**2) Coluna "Age":**

#**Verificação: Distribuição Normal**
# Para uma possível substituição dos valores nulos pela média de idade na coluna "Age":

#**Histograma:**

# Verificando se há uma distribuição normal com a coluna de idade
plt.hist(dataset_titanic_train["Age"], histtype = "stepfilled", rwidth = 0.8)
plt.xlabel("Idade")
plt.ylabel("Frequência")

#**KDEPlot:**

# Verificando se há uma distribuição normal com a coluna de idade
sea.kdeplot(data=dataset_titanic_train["Age"], shade=True)

# Justificativa: Substituição de valores nulos pela média de idade em razão da distribuição de dados desta coluna apresentar uma distribuição normal e, além disso, com a presença de poucos outliers

# Média de idade
filter1_mean = dataset_titanic_train["Age"].mean()

# Substituindo na coluna "Age"
dataset_titanic_train.loc[dataset_titanic_train.Age.isnull(),"Age"] = filter1_mean

# Verificação da substituição
dataset_titanic_train.info()

#**3) Substituição dos valores nulos no "Embarked" pela moda:

# Justificativa:** Moda é pouco sensível a outliers e, além disso, há poucos valores únicos.

# Moda na coluna Embarked
filter2_mode = dataset_titanic_train["Embarked"].mode()[0]
filter2_mode

# Substituição dos NaN pela moda
dataset_titanic_train.loc[dataset_titanic_train.Embarked.isnull(),"Embarked"] = filter2_mode

# Verificação da substituição
dataset_titanic_train.info()

#**4) KNNImputer: outra abordagem para valores nulos**

from sklearn.impute import KNNImputer

# X = dataset_titanic_train.drop(columns=['Age', 'Sex', 'Embarked'])
# imputer = KNNImputer(n_neighbors=15)
# predict = imputer.fit_transform(X)
# predict

#**Tratamento de Dados (test.csv): Parte I**

# Colunas com valores nulos
dataset_titanic_test.isnull().sum()

#**1) Eliminação das colunas "Name", "Ticket" e "Cabin":**

# Justificativa:** Alta cardinalidade

# Eliminação das colunas abaixo pela alta cardinalidade
dataset_titanic_test = dataset_titanic_test.drop(["Name","Ticket","Cabin"],axis=1)

# Verificação da exclusão das colunas
dataset_titanic_test.columns

#**2) Substituição de valores nulos pela média de idade na coluna "Age":**

# Justificativa:** Substituição de valores nulos pela média de idade em razão da distribuição de dados desta coluna apresentar uma distribuição normal e, além disso, com a presença de poucos outliers

# Média de idade
filter1_mean = dataset_titanic_test["Age"].mean()
filter1_mean

# Substituindo na coluna "Age"
dataset_titanic_test.loc[dataset_titanic_test.Age.isnull(),"Age"] = filter1_mean

# Verificação da substituição
dataset_titanic_test.info()

#**3) Substituição dos valores nulos pela moda na coluna "Embarked":**

# Não precisa pois apresenta 0 valores nulos

#**4) Substituição dos valores nulos pela mediana na coluna "Fare":**

# Justificativa:** Mediana foi aplicada em razão da distribuição não normal em tal coluna e devido ao grande número de outliers.

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

#**Tratamento de Dados (train.csv): Parte II**

#**1) Substituição de 1 para male e 0 para female na coluna "Sex":**

# Justificativa:** Aumenta o número de atributos para treinamento do modelo.

# Dataset de treino: substituição
dataset_titanic_train.Sex = dataset_titanic_train.Sex.replace(["female"], "0")
dataset_titanic_train.Sex = dataset_titanic_train.Sex.replace(["male"], "1")
dataset_titanic_train.Sex

dataset_titanic_train['Sex'] = dataset_titanic_train['Sex'].astype(int)
dataset_titanic_train.Sex.dtype

#**2) Aplicação do OneHotEncoder na coluna "Embarked":**

# A aplicação desta ferramenta possiblilita mais um atributo ao aprendizado do modelo:**

# Import
from sklearn.preprocessing import OneHotEncoder

# Creating Encoder
ohe = OneHotEncoder(handle_unknown="ignore")

# Data Fit
ohe = ohe.fit(dataset_titanic_train[["Embarked"]])

# Tranforming
ohe.transform(dataset_titanic_train[["Embarked"]]).toarray()

# Changing to a DataFrame
ohe_df = pd.DataFrame(ohe.transform(dataset_titanic_train[["Embarked"]]).toarray(),columns=ohe.get_feature_names_out())

# Concatening ohe_df
dataset_titanic_train = pd.concat([dataset_titanic_train, ohe_df], axis=1)

# Verificação
dataset_titanic_train.head()

# Exclusão da coluna "Embarked":
dataset_titanic_train = dataset_titanic_train.drop("Embarked", axis=1)
dataset_titanic_train

#**Tratamento de Dados (test.csv): Parte II**

#**1) Substituição de 1 para male e 0 para female na coluna "Sex":**

# Justificativa: Aumenta o número de atributos para treinamento do modelo.

# Dataset de treino: substituição
dataset_titanic_test.Sex = dataset_titanic_test.Sex.replace(["female"], "0")
dataset_titanic_test.Sex = dataset_titanic_test.Sex.replace(["male"], "1")
dataset_titanic_test.Sex

dataset_titanic_test['Sex'] = dataset_titanic_test['Sex'].astype(int)
dataset_titanic_test.Sex.dtype

#**2) Aplicação do OneHotEncoder na coluna "Embarked":**

# A aplicação desta ferramenta possiblilita mais um atributo ao aprendizado do modelo:**

# Import
from sklearn.preprocessing import OneHotEncoder

# Creating Encoder
ohe = OneHotEncoder(handle_unknown="ignore")

# Data Fit
ohe = ohe.fit(dataset_titanic_test[["Embarked"]])

# Tranforming
ohe.transform(dataset_titanic_test[["Embarked"]]).toarray()

# Changing to a DataFrame
ohe_df = pd.DataFrame(ohe.transform(dataset_titanic_test[["Embarked"]]).toarray(),columns=ohe.get_feature_names_out())

# Concatening ohe_df
dataset_titanic_test = pd.concat([dataset_titanic_test, ohe_df], axis=1)

# Verificação
dataset_titanic_test.head()

# Exclusão da coluna "Embarked":
dataset_titanic_test = dataset_titanic_test.drop("Embarked", axis=1)
dataset_titanic_test

#**Preparação dos Dados:**

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

#**Regressão Logística:**

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

#**KNN:**

from sklearn.neighbors import KNeighborsClassifier

# Criação do Classificador
clf_knn = KNeighborsClassifier(n_neighbors=3)

# Data Fit
clf_knn = clf_knn.fit(X_train,y_train)

# Previsão
y_pred_knn = clf_knn.predict(X_val)

#**Avaliação do Modelo:**

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Accuracy: Regressão Logística
accuracy_rl = accuracy_score(y_val,y_pred_rl)

# Accuracy: Árvore de Decisão
accuracy_tree = accuracy_score(y_val,y_pred_tree)

# Accuracy: KNN
accuracy_knn = accuracy_score(y_val,y_pred_knn)

# Matriz de Confusão : Regressão Logística
confusion_matrix(y_val, y_pred_rl)

# Matriz de Confusão: Árvore de Decisão
confusion_matrix(y_val, y_pred_tree)

# Matriz de Confusão: KNN
confusion_matrix(y_val, y_pred_knn)

#**Tabela de performance:**

performance = pd.DataFrame({
    "Modelos": ["Regressão Logística","Árvore de Decisão","KNN"],
    "Acurácia": [accuracy_rl, accuracy_tree, accuracy_knn]
})
performance

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
dataset_final.to_csv("dataset_titanic_2_final.csv",index=False)
dataset_final
