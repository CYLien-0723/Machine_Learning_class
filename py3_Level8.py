# -*- coding: utf-8 -*-
"""
(a)教安裝pytouch
    (py3-8 00:50:00)    
    (https://pytorch.org/)
    請研究>>(C:/python3_class/Level8/1.PyTorch – Implementing First Neural Network-Softmax-batch_size-MNIST.ipynb)
    
(b)較安裝RAdam
    (https://github.com/CyberZHG/keras-radam/blob/master/README.zh-CN.md)
    請研究>>(C:/python3_class/Level8/day8-keras-cnn-start-part1-no-maxpooling-radam-tensorflow-pending.ipynb)

(c)RNN基本上是針對跟時間序列相關的DATA(如:語言翻譯/股票?...之類的):
    下面還是用RNN示範一次py3_Level2-test1的例子(也就是數子的例子)
"""
#=====================用RNN做數字的深度學習====================
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.model_selection import train_test_split 
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN
(X_train, y_train), (X_test, y_test) = mnist.load_data()

x1 = X_train.astype('float32') / 255.
y1 = np_utils.to_categorical(y_train)
x2 = X_test.astype('float32') / 255.
y2 = np_utils.to_categorical(y_test)   

#------------建立神經層--------------
n_in, n_h, n_out = 28*28, 50, 10 #四個特徵值、隱藏層有五個神經元、預計輸出有三種類別
model = Sequential()
model.add(SimpleRNN(units=400, activation='relu', input_shape=(28,28)))
model.add(Dense(units=n_out, 
                kernel_initializer='normal',
                activation='softmax'))
# g, no. of FFNNs in a unit (RNN has 1, GRU has 3, LSTM has 4) >>所以在這邊是1
# h, size of hidden units >>在這邊是400
# i, dimension/size of input >>在這邊是28
#RNN的參數算法fnum_params = g × [h(h+i) + h]
print(model.summary())

#------------建立模組(優化器..)--------------
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd',
              metrics=['accuracy'])
train_history=model.fit(x=x1.reshape(-1,28,28),#注意 使用 RNN需要轉換 維度
                        y=y1,validation_split=0.2, 
                        epochs=10, batch_size=50,verbose=1)

print(model.weights)# 顯示 model

#------------驗證精準度--------------
# 驗證Pytorch針對Test資料集的驗證
#x_test = torch.from_numpy(X_test).float()   
#y_test_pred = model(x_test.to(device))
scores = model.evaluate(x1.reshape(-1,28,28), y1)
print('accuracy x1 =',scores[1])
scores = model.evaluate(x2.reshape(-1,28,28), y2)
print('accuracy x2=',scores[1])


#------------畫圖--------------
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#------------預測--------------
from skimage import io 
import numpy as np
import matplotlib.pyplot as plt
my_image = io.imread('C:/python3_class/Level6/MysteryNumberD.bmp', as_gray=True)
plt.figure(figsize=(1,1))
plt.axis('off')
plt.imshow(my_image, cmap='gray_r')
my_image.shape
np.argmax(model.predict(my_image.reshape(1, 28, 28).astype('float32')))