# -*- coding: utf-8 -*-
'''
(一)#import os
    #os.environ['KERAS_BACKEND']='theano'
    #import keras#需要上面兩行(原因如右:https://morvanzhou.github.io/tutorials/machine-learning/keras/1-3-backend/)

(二)#===================用ML的方式處理數字辨識=======================
    #可參考 https://keras.io/examples/mnist_cnn/ 裡面的作法
    #可參考 https://keras.io/optimizers/ 裡面有多種優化器(模型)
    #0維稱純量[1] / 1維稱向量[1,2,3] / 2維~N維統稱張量(tensor)
    #而在GOOGLE的框架底下稱tensorflow(py3-3 2:28:00有解釋)

(三)一些np用法
    (a)np.hstack用法如下:
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        res = np.hstack((arr1, arr2))
        結果為>>[1 2 3 4 5 6]
        arr1 = np.array([[1, 2], [3, 4], [5, 6]])
        arr2 = np.array([[7, 8], [9, 0], [0, 1]])
        res = np.hstack((arr1, arr2))
        結果為>>[[1 2 7 8]
                 [3 4 9 0]
                 [5 6 0 1]]
    
    (b)np.vstack用法如下:
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        res = np.vstack((arr1, arr2))
        結果為>>array([[1, 2, 3],
                    [4, 5, 6]])
'''

#==========1)取得資料庫
import matplotlib.pyplot as plt
from sklearn import datasets
mnist = datasets.load_digits()
mnist.keys()

#==========2)將資料分成訓練跟測試
from keras import datasets
datasets.mnist.load_data? #>>可以查看需要的引述與回傳值有哪些
datasets.mnist.load_data? #>>可以查看需要的引述與回傳值有哪些
datasets.mnist.load_data? #>>可以查看需要的引述與回傳值有哪些
print(dir(datasets.mnist))#>>可以看有哪些方法可用
print(dir(datasets.mnist))#>>可以看有哪些方法可用
print(dir(datasets.mnist))#>>可以看有哪些方法可用
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape))
#plt.figure(figsize=(1,1))
#plt.axis('off')
#plt.imshow(x_train[0],cmap='gray_r')
#plt.show()

#==========3)檢視資料(也可以說成是在做標準化>>但下面我們是用專門處理影像事會做的方式)
import numpy as np
#--a)將x_train裡面的數值從0~255變成0~1
print(np.min(x_train[0]),np.max(x_train[0]),x_train[0].dtype)#發現max, min 建議要改成float32
x_train_32 = x_train.astype('float32')/255.
x_test_32 = x_test.astype('float32')/255.
'''
(一)再py3-4 2:51:00 將feature多加np.square()>>也就是將裡面每個數值等平方後>>會將準確率提升
    x_train_32 = np.square(x_train.astype('float32')/255.)
    x_test_32 = np.square(x_test.astype('float32')/255.)

(二)模擬CNTK如下的做法在keras怎麼實現(請py3-5 1:12:52)>>這是DAT236x Ch2 的Question6
	r=C.laryers.Dense(num_output= )(C.splice(featrues,C.sqrt(feartures),C.square(feartures),axis=0))
	>>他這個做法就是把原始feature / 經過sqrt處理過的feature / 經過square處理過的feature 並再一起
	>>也就是原本一個圖為(784,)變成(784*3,)
    要改的地方如下:
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
    model.add(Dense(10, input_shape=(784*3,), activation='softmax', kernel_initializer=('glorot_uniform') ))
    其他地方都一樣
'''

#--b)將x_train裡面的資料從(60000,28,28)轉成(60000,784)
x_train_32 = x_train_32.reshape(60000,784)
x_test_32 = x_test_32.reshape(10000,784)
print(x_train_32.shape,x_test_32.shape)

#--c)將y_train裡面的資料轉成one-hot encoding
#(方式一)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train_oh = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test_oh = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
#<註>ohe.fit_transform(y_train)>>會顯示error
#    >>Expected 2D array, got 1D array instead 
#    >>所以要用y_train.reshape(-1,1)解決
#<註>y_train.shape>>會顯示(6000,)   
#<註>y_train.reshape(-1,1).shape>>會顯示(6000,1)

#(方式二)
import keras
y_train_oh= keras.utils.to_categorical(y_train, 10)#10>>因為我們數值種類只有10種
y_test_oh= keras.utils.to_categorical(y_test, 10)#10>>因為我們數值種類只有10種
print(y_train_oh.shape,y_test_oh.shape)


#==========4)建造深度學習model(輸入為784個feature的 1D / 128層神經層 / 輸出為10個種類OR答案)
from keras.models import Sequential #>>第一層
from keras.layers import Dense #>>神經層()
#--a)建立模型
#可參考 https://keras.io/optimizers/ 裡面有多種優化器(模型)
model = Sequential()
#(第一層神經層)輸入為784個feature的一維(1D)資料,此神經層由128節點組成
model.add(Dense(128, input_shape=(784,),activation='relu'))
#(第二層神經層)softmax意思是:輸出為10個種類,將每個種類中機率最大是為答案
model.add(Dense(10, activation='softmax'))

'''
<<觀念一>>
  如果只要輸入and輸出(不要中間的神經層>>也就是只用一個神經層)>>則如下
  model.add(Dense(10, input_shape=(784,),activation='softmax')
  (共一層神經層)總共運算次數為>>784*10+10(wx+b)

  如果是要多個且多種神經層>>則如下
  model.add(Dense(400, input_shape=(784,),activation='relu')
  model.add(Dense(200,activation='relu')
  model.add(Dense(10,activation='softmax')
  (共三層神經層)總共運算次數為>>784*400+400(wx+b) + 200*400+200(wx+b) + 200*10+10(wx+b)
<<觀念二>>
  中間的神經層會用到一個"啟發函數 activation function"(ex:relu,sigmoid...)
'''
model.compile(loss=keras.losses.categorical_crossentropy, #偏差值function的設定
              optimizer=keras.optimizers.SGD(lr=0.2), #優化器(運算的模型)>>用甚麼優化器就要記得import / lr>學習率
              metrics=['accuracy'])
print(model.summary())#會print出下面的資料
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
None

100480>>是784*128+128而來
101779>>是(784*128+128)+(784*128+784)+(128*10+10)而來
理論上 101779(Total params)越大成效越好，但也不一定
'''
#--b)訓練模型
result = model.fit(x_train_32 , y_train_oh,#要給的訓練資料
          batch_size=64,#每次訓練時採用的樣本個數(batch_size=1表示只採用1個,batch_size=Full Batch Learning表示用全部>>這邊是依照Dat236x給的64)
          epochs=40,#epoch所代表的數字是指所有數據被訓練的總輪數
          verbose=1,#0>>不顯示訊息 / 1>>immediate顯示訊息 / 2>>buffer顯示訊息
          validation_split=0.2)#validation_split>>用剛剛的訓練資料去驗證  / validation_data>>另外再給驗證資料

#--c)將訓練模型的過程其loss與accuray值畫成圖
print(result.history)
#利用result.history()>>會回傳一個dict其裡面有4個欄位[val_loss,val_acc,loss,acc]
#前兩個欄位[val_loss,val_acc]是驗證(val)的資料,後兩個欄位[loss,acc]是訓練資料的
#其row(也就是每個欄位的資料個數)是看你epochs是用多少
import matplotlib.pyplot as plt
plt.plot(result.history['acc'],color='blue',label='acc')
plt.plot(result.history['val_acc'],color='red',label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.show()#如果圖red與blue的現都是"一起往上增加"才是好的模型

plt.plot(result.history['loss'],color='blue',label='acc')
plt.plot(result.history['val_loss'],color='red',label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.show()#如果圖red與blue的現都是"一起往下減少"才是好的模型

#--d)預測資料(驗證結果)
#用tensorflow VS cntk 準確率會有差>>在(C:\Users\ben\.keras\keras.json>>"backend": "xxxx")地方更改xxxxx為tensorflow 還是 cntk
#安裝cntk==2.6一值error>>待解決

#(方式一)>> y_test_oh要是one-hot encoding
score = model.evaluate(x_test_32 , y_test_oh,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

#(方式二)>> y_test要是array
print(np.sum(y_test[:]==model.predict_classes(x_test_32[:]))/len(y_test))

#--e)列出前25筆答案與模組預測出來的答案
print(y_test[:25])
print(model.predict_classes(x_test_32[:25]))



