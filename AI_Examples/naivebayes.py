# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:33:24 2023

@author: Dnaiel Nogueira

References
https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
https://www.v7labs.com/blog/f1-score-guide

Tipos de Classificação 
- Gaussian Naive Bayes (GaussianNB): 
    Este modelo assume que as características têm uma distribuição 
    gaussiana (normal). É apropriado quando seus dados contínuos 
    se aproximam de uma distribuição normal.
- Multinomial Naive Bayes (MultinomialNB): 
    Este modelo é usado quando as características são contagens discretas,
    como contagens de palavras em um texto. É amplamente utilizado em tarefas
    de classificação de texto.
- Bernoulli Naive Bayes (BernoulliNB): 
    Este modelo é adequado para dados binários, onde as características
    são representadas como valores binários (0 ou 1). É frequentemente 
    usado em tarefas de classificação de documentos binários, 
    como detecção de spam.
- Complement Naive Bayes (ComplementNB): 
    Este é uma variação do Multinomial Naive Bayes que é projetada para
    conjuntos de dados desequilibrados. Ele ajusta os parâmetros de 
    probabilidade para tratar de maneira mais eficaz classes sub-representadas.
- Categorical Naive Bayes (CategoricalNB): 
    Este modelo é usado para variáveis categóricas, em que as características
    são representadas como categorias discretas. É útil em tarefas de
    classificação com dados categóricos.

"""
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Read dataset from a CSV file
dataset = pd.read_csv('iris.csv', header=None, sep=',')
dataset.columns =['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']

# Find categorical variables
classes = np.unique(dataset["Class"].values)
categorical = [var for var in classes]
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

# Split dataset - Inputs (x) and Outputs (y)
x = dataset[['Sepal length', 'Sepal width', 'Petal length', 'Petal width']].values
y = dataset[['Class']].values
y = y.reshape(-1,)

# Split dataset - train and test
X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=125)

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict(X_test)
n = 6
print("Actual Value:", y_test[n])
print("Predicted Value:", predicted[n])

result = pd.DataFrame(columns=["Real", "Predicted"])
result['Real'] = y_test
result['Predicted'] = predicted
result['Evaluate'] = result['Real'] == result['Predicted']
print(result)

# Model Evaluation
mislabeled = (y_test != predicted)
print("Number of mislabeled points out of a total %d points : %d"%(len(y_test),
                                                                   mislabeled.sum()))
accuray = accuracy_score(predicted, y_test)
recall = recall_score(y_test, predicted, average="weighted")
f1 = f1_score(predicted, y_test, average="weighted")
print("Accuracy:", accuray)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
labels = np.unique(y_test)
cm = confusion_matrix(y_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

recallSetosa,recallVersicolor,recallVirginica = recall_score(y_test, predicted, average=None)
print("Recall - Setosa:", recallSetosa)
print("Recall - Versicolor:", recallVersicolor)
print("Recall - Virginica:", recallVirginica)

f1Setosa,f1Versicolor,f1Virginica = f1_score(predicted, y_test, average=None)
print("F1 score - Setosa:", f1Setosa)
print("F1 score - Versicolor:", f1Versicolor)
print("F1 score - Virginica:", f1Virginica)








