# -*- coding: utf-8 -*-
"""
Created on Sat May 27 06:03:41 2023

@author: daniel nogueira

"Boston Housing":

Atributos (Features):

Existem 13 atributos (ou características) que descrevem diferentes aspectos das
casas na área de Boston. Estes incluem parametros como: taxa de criminalidade, 
proporção de hectares de terrenos residenciais, proximidade com o rio Charles,
entre outros. Cada exemplo (instância) no conjunto de dados é uma casa descrita
por esses 13 atributos.

Variável de Destino (Target):

A variável de predição é o valor médio das casas ocupadas pelos proprietários 
em milhares de dólares. Esta variável é frequentemente chamada de "MEDV" (Median Value).

Divisão entre Conjunto de Treinamento e Teste:

Os dados são frequentemente divididos em um conjunto de treinamento e um 
conjunto de teste. No Keras, a função boston_housing.load_data() retorna dois 
conjuntos de dados: (x_train, y_train) para treinamento e (x_test, y_test) para teste.

Dimensionalidade:

Cada exemplo no conjunto de dados consiste em 13 valores para os atributos 
e um valor para a variável de destino. Portanto, a entrada x_train (e x_test) 
é uma matriz com forma (número_de_exemplos, 13), e a variável de destino 
y_train (e y_test) é uma matriz com forma (número_de_exemplos,).

Escalas Diferentes:

Alguns atributos podem ter escalas diferentes. Em muitos casos, é benéfico 
normalizar ou padronizar os dados antes de treinar um modelo.

The Boston Housing Dataset

O Boston Housing Dataset é derivado de informações coletadas pelo 
U.S. Census Service em relação à habitação na área de Boston MA.

CRIM    - taxa de criminalidade per capita por cidade
ZN      - proporção de terrenos residenciais zoneados para lotes acima de 
          25.000 pés quadrados.
INDUS   - proporção de hectares comerciais não varejistas por cidade.
CHAS    - Variável dummy Charles River (1 se o trato limita o rio; 0 caso contrário)
NOX     - concentração de óxidos nítricos (partes por 10 milhões)
RM      - número médio de quartos por alojamento
IDADE   - proporção de unidades ocupadas pelos proprietários construídas antes de 1940
DIS     - distâncias ponderadas para cinco centros de emprego de Boston
RAD     – índice de acessibilidade às rodovias radiais
IMPOSTO - taxa de imposto sobre a propriedade de valor total por US$ 10.000
PTRATIO - proporção aluno-professor por cidade
B       - 1000(Bk - 0,63)^2 onde Bk é a proporção de negros por cidade
LSTAT   - % status mais baixo da população
MEDV    - Valor médio das casas ocupadas pelos proprietários em US$ 1.000
​

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

# Carregamento do conjunto de dados Boston Housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Normalização dos dados
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Criação do modelo da CNN
modelo = Sequential()
# Adição da primeira camada densa (totalmente conectada) com 100 neurônios e função de ativação ReLU
modelo.add(Dense(100, input_dim=x_train.shape[1], activation='relu', name='Hidden-1'))
# Adição da segunda camada densa com 100 neurônios e função de ativação ReLU
modelo.add(Dense(100, activation='relu', name='Hidden-2'))
# Adição da camada de saída com 1 neurônio (regressão) e função de ativação linear
modelo.add(Dense(1, activation='linear', name='Output'))

# Compilação do modelo usando o otimizador 'adam' e a função de perda 'mean_squared_error' para problemas de regressão
modelo.compile(optimizer='adam', loss='mean_squared_error')

'''
1. Criação do Modelo Sequencial (Sequential):
    Inicializa um modelo sequencial. 
    Um modelo sequencial é apropriado para uma pilha linear de camadas, 
    onde cada camada tem exatamente um tensor de entrada e um tensor de saída.

2. Adição da Primeira Camada Densa (Dense):
    Adiciona a primeira camada densa com 100 neurônios, função de ativação ReLU
    e especifica a dimensão de entrada como x_train.shape[1]. Isso define 
    a camada de entrada da rede.
    
3. Adição da Segunda Camada Densa (Dense):
    Adiciona a segunda camada densa com 100 neurônios e função de ativação ReLU.
    Isso cria uma camada intermediária na rede.

4. Adição da Camada de Saída (Dense):
    Adiciona a camada de saída com 1 neurônio (comum em problemas de regressão)
    e função de ativação linear.

5. Compilação do Modelo (compile):
    Configura o modelo para treinamento, especificando o otimizador 'adam' e a 
    função de perda 'mean_squared_error'. A escolha de 'mean_squared_error' 
    como a função de perda indica que o modelo está sendo treinado para minimizar
    o erro quadrático médio, adequado para problemas de regressão onde o objetivo 
    é prever valores numéricos.
'''
# Treinamento do modelo
hist = modelo.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Avaliação do modelo
loss = modelo.evaluate(x_test, y_test)
print('Loss:', loss)


figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()


# Predição
pred = modelo.predict(x_test)
y_pred = np.squeeze(pred, axis=1)

result = pd.DataFrame(y_test, columns=['Real'])
result['Pred'] = y_pred

plt.figure(figsize=(10, 8))
plt.plot(y_test,y_test,'k')
plt.scatter(y_test,y_pred)
plt.xlabel('Valor Real')
plt.ylabel('Valor Predito')

plt.figure(figsize=(15, 15))
plt.plot(y_test, 
         linestyle='--', 
         marker='D', 
         color='b', label='Real')
plt.plot(y_pred, 
         #marker='o', 
         color='r', label='Pred')
plt.legend()
plt.show()

model2 = Sequential()
model2.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(1))

model2.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae'])


# Treinamento do modelo
hist2 = model2.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Avaliação do modelo
loss2 = model2.evaluate(x_test, y_test)
print('Loss:', loss2)


figure = plt.gcf()
figure.set_size_inches((15, 5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, len(hist2.history['loss']) + 1), hist2.history['loss'])
plt.plot(range(1, len(hist2.history['val_loss']) + 1), hist2.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()


# Predição
pred2 = model2.predict(x_test)
y_pred2 = np.squeeze(pred2, axis=1)

result['Pred2'] = y_pred2

plt.figure(figsize=(10, 8))
plt.plot(y_test,y_test,color='black', linestyle='--')
plt.scatter(y_test,y_pred,label='Predição MODELO')
plt.scatter(y_test,y_pred2,label='Predição MODEL2', color='red', marker='^')
plt.xlabel('Valor Real')
plt.ylabel('Valor Predito')

plt.figure(figsize=(15, 15))
plt.plot(y_test, 
         linestyle='--', 
         marker='D', 
         color='b', label='Real')
plt.plot(y_pred, 
         #marker='o', 
         color='r', label='Pred')
plt.legend()
plt.show()
