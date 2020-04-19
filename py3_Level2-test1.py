# -*- coding: utf-8 -*-
"""
test為純跑code的範例
沒注釋沒標註
"""

import matplotlib.pyplot as plt
from sklearn import datasets
mnist = datasets.load_digits()
mnist.keys()

from keras import datasets
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape))

import numpy as np
print(np.min(x_train[0]),np.max(x_train[0]),x_train[0].dtype)
x_train_32 = np.hstack([x_train.astype('float32') / 255., 
                        np.sqrt(x_train.astype('float32') / 255.),
                        np.square(x_train.astype('float32') / 255.)
                      ])
x_test_32 = np.hstack([x_test.astype('float32') / 255., 
                       np.sqrt(x_test.astype('float32') / 255.),
                       np.square(x_test.astype('float32') / 255.)
                      ])
x_train_32 = x_train_32.reshape(60000, 784*3)
x_test_32 = x_test_32.reshape(10000, 784*3)
print(x_train_32.shape, x_test_32.shape)

import keras
y_train_oh= keras.utils.to_categorical(y_train, 10)
y_test_oh= keras.utils.to_categorical(y_test, 10)
print(y_train_oh.shape,y_test_oh.shape)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape=(784*3,), activation='softmax', kernel_initializer=('glorot_uniform') ))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(0.2),
              metrics=['accuracy'])
print(model.summary())

result = model.fit(x_train_32 , y_train_oh,
          batch_size=64,
          epochs=10,
          verbose=1,
          validation_split=0.2)

print(result.history)

import matplotlib.pyplot as plt
plt.plot(result.history['acc'], color='red')
#plt.plot(result.history['val_acc'], color='red')
plt.grid()
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.show()

plt.plot(result.history['loss'], color='blue')
#plt.plot(result.history['val_loss'], color='red')
plt.grid()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

score = model.evaluate(x_test_32, y_test_oh, batch_size=512 , verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Average test error:', (1 - score[1]) * 100, '%')


print( y_test[:25] )
print(model.predict_classes(x_test_32[:25]))



