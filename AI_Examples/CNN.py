# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 21:27:00 2023

@author: Daniel Nogueira

#x_train /= 255
#x_test /= 255
#from keras.utils import to_categorical
#y_train = to_categorical(y_train, num_classes)
#y_test = to_categorical(y_test, num_classes)


"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.datasets import mnist
from keras.models import Sequential#, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
#import tensorflow as tf 


##################### DATASET #########################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_index = 10
print("Imagem é de valor %i"%y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()
print(x_train.shape)
print(x_test.shape)

numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# save input image dimensions
img_rows, img_cols = x_train[0].shape    #  28, 28

# redimensionamento das imagens para um valor padrão
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

num_classes = len(np.unique(y_train)) #10

##################### MODEL #########################

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
     activation='relu',
     input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])


##################### TRAIN #########################
batch_size = 128
epochs = 10

training_history = model.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             #validation_data=(x_test, y_test),
                            verbose=1)

score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['loss'], label='training set')
#plt.plot(training_history.history['val_loss'], label='validation set')
plt.legend()
plt.show()

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
#plt.plot(training_history.history['val_accuracy'], label='validation set')
plt.legend()
plt.show()
##################### TEST #########################
y_pred = model.predict(x_test)

pred = np.round(y_pred)

n = 115
ind = np.where(pred[n]==1)[0][0]

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

num_classes = 10
#y_train = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

ind_test = np.where(y_test_cat[n]==1)[0][0]

print("The predicted value is %i and the real value is %i"%(ind, ind_test))

model.save("model.h5")
model.save_weights("model_weights.h5")

pred = np.argmax(y_pred, axis=1)

df = pd.DataFrame(y_test, columns=['Real'])
df['Pred'] = pred

diferentes = (df['Real'] != df['Pred']).sum()

print(f'O número de linhas com valores diferentes é: {diferentes}')

df['Diferenca'] = df['Real'] != df['Pred']
linhas_com_diferenca = df[df['Diferenca']]

image_index = linhas_com_diferenca.index[1]
plt.imshow(x_test[image_index], cmap='Greys')
plt.show()

numbers_to_display = 272
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15, 15))
for plot_index in range(numbers_to_display): 
     predicted_label = pred[plot_index]
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     color_map = 'Greens' if predicted_label == y_test[plot_index] else 'Reds'
     plt.subplot(num_cells, num_cells, plot_index + 1)
     plt.imshow(x_test[plot_index].reshape((img_rows, img_cols)), cmap=color_map)
     plt.xlabel(predicted_label)
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(y_test, pred)

# Exibindo a matriz de confusão usando seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusão')
plt.show()