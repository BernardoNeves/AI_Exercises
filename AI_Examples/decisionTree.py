# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:21:52 2023

@author: Daniel Nogueira

References:
    
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html


"""

from sklearn.tree import DecisionTreeClassifier
import graphviz 
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
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
features_cols = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
label = ['Class']
x = dataset[features_cols].values
y = dataset[label].values
y = y.reshape(-1,)

# Split dataset - train and test
X_train, X_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=125)

# Building Decision Tree
'''
Cria um classificador de árvore de decisão com alguns parâmetros. 
- O parâmetro criterion define a medida de impureza usada para dividir os nós 
da árvore (neste caso, "entropia"). 
- max_depth define a profundidade máxima da árvore
- min_samples_leaf especifica o número mínimo de amostras necessárias em um nó folha.
'''
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=4, 
                                  min_samples_leaf=4)
# criterion defines the impurity measure used to split the nodes of the tree ("entropy")
# max_depth represents max level allowed in each tree 
# min_samples_leaf minumum samples storable in leaf node

# Fit the tree to iris dataset
clf.fit(X_train, y_train)

# Predict Output
predicted = clf.predict(X_test)
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

# Plot decision tree
'''
Gera uma representação em formato DOT da árvore de decisão treinada. 
Isso é feito com o auxílio da função export_graphviz da biblioteca tree. 
Os parâmetros especificam: nomes de recursos, nomes de classes e que o gráfico 
gerado deve ser preenchido, arredondado e conter caracteres especiais.
'''
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=features_cols,  
                     class_names=labels,  
                     filled=True,
                     rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

##############################################################################
clf2 = tree.DecisionTreeClassifier(criterion='gini',
                                   max_depth=4,
                                   min_samples_leaf=4)

clf2.fit(X_train, y_train)
dot_data2 = tree.export_graphviz(clf2, out_file=None,
                                 feature_names=features_cols,  
                                 class_names=labels,  
                                 filled=True, 
                                 rounded=True,  
                                 special_characters=True)  
graph2 = graphviz.Source(dot_data2)  
graph2 
