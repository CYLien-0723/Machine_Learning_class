# -*- coding: utf-8 -*-
"""
()join與astype小妙用
    輸入 >> ' '.join(y_train_oh[0].astype('int').astype('str'))
    就可以把y_train_oh[0]先轉成int再轉成str在接到' '的字串後面

()寫入txt的方式:
    with open('C:/python3_class/Level7/mnist.train.txt', 'w') as f:
        for i in range(X_train.shape[0]):
            f.write('|labels ' + ' '.join(y_train_oh[i].astype('int').astype('str')) +'\n') 

卷積神經網絡(Convolutional Neural Network)簡稱CNN

(a)CNN的原理(可看py3-6 01:30:00):
    將一張圖(data),假設為100*100
    (一)我們先用一些moduel做影像處理(ex:銳化/翻轉...等等)來得到多張圖:
        如此一張圖就變成多張圖(多個經過特殊處理的data)
    (二)將多張圖的data結合在一起:
        原本一張圖為100*100>>經過n種處理
        則此data就變成100*100*n
    (三)將經過多種處理的每張圖縮小:
        100*100*n*(1/X)
    原本ML(Multilayer) Perceptron 輸入為[10000]>>[400]>>[10]假設為10種動物要辨認
    但因為上面影像處理過後產的多張圖 所以輸入便為[10000*n*(1/X)]>>[400]>>[10]

(b)影像處理用CNN比MLP好的>>三種好處(最主要就是要運算的參數變少 )
    (一)Preserve the spatial structure of the input data(保持空間結構)
    (二)Reduce the number of model parameters(同樣的神經層>>需要運算的參數較少)
    (三)Detect features irrespective of the location in the image(圖像特徵被保留)
    
(c)教keras與cntk兩邊的code的比較與轉換>>因為本人懶得在key cntk與用cntk的code,所以這邊不解釋
    可去看py3-7 02:00:00-02:40:00的影片

**參數多寡影響速度/記憶體/也影像是否overfitting**
**參數多寡影響速度/記憶體/也影像是否overfitting**
**參數多寡影響速度/記憶體/也影像是否overfitting**

"""

#======================開始實際練習CNN=====================
from keras import datasets
from keras.utils import to_categorical
import numpy as np
#將資料庫load進來
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

#將x做標準化(*1/255)並reshape成4維
#原本為(60000,28,28)>>但因為圖都要在加個Channel>>所以變為(60000,28,28,1)
X_train =  (X_train.astype(np.float32)/255.0).reshape(X_train.shape[0], 28, 28, 1)
X_test =  (X_test.astype(np.float32)/255.0).reshape(X_test.shape[0], 28, 28, 1)

#將y做one-hot encoding
y_train_oh = to_categorical(y_train, num_classes=10, dtype='float32')
y_test_oh = to_categorical(y_test, num_classes=10, dtype='float32')

#建立神經層
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D , MaxPooling2D#(有分MaxPooling1D/MaxPooling2D/MaxPooling3D), AveragePooling2D
from keras import backend as K
#下面的判斷式>>實際上可以直接點開C:\Users\Ben\.keras\keras.json看裡面的image_data_format是寫哪一個
if K.image_data_format() == 'channels_first': input_shape = (1, 28, 28)
elif K.image_data_format() == 'channels_last': input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(filters=8,input_shape=input_shape,kernel_size=(5, 5),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=25,kernel_size=(3, 3),strides=(1,1),padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())#把25張圖攤平>>類似[[1,1],[2,2]]+[[3,3],[4,4]] >>變成[1,1,2,2,3,3,4,4]
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
print(model.summary())
'''
**CNN的參數算法**(搭配model.summary()看)**
第一層(convolutional layer)的conv2d_2 (Conv2D)  
    Param=650 >> 是(5*5*1+1)*8 = 208而來
    其中:5*5為kernel_size=(5, 5) / *1為(28,28,1)的1,如果是3也就是RGB各一張圖,但這邊因為圖只有灰階所以為1>>用3的話會下面跑error
         +1為bias / *25為filters=25
    ###動畫詮釋convolutional layer在幹嘛>>https://www.cntk.ai/jup/cntk103d_conv2d_final.gif
第二層(pooling layers) 中pool_size=(2,2)是將2*2取最大值,縮成取一格
    會多加這層主要是為了防止overfitting
    (28,28,25)>>經過MaxPooling2D(pool_size=(2,2))>>變成(14,14,25)
    (28,28,25)>>經過MaxPooling2D(pool_size=(2,2),strides=(1,1))>>變成(27,27,25)
第三層(convolutional layer)的conv2d_2 (Conv2D)  
    Param=1825 >> 是(8*3*3+1)*25而來
    其中:8*為前一層的filters / 3*3為kernel_size=(3, 3) / +1為bias / *25為filters=25
第四層(pooling layers)中pool_size=(2,2),strides=(2,2) 將 (14, 14, 25) 變成 (7, 7, 25)
    AveragePooling2D>>取平間值
    MaxPooling2D>>取最大值
第五層的flatten_2 (Flatten)  Output_Shape=XXXXXX >>是XX*XX*25而來>>顧名思義就是flatten得意思
第六層的dense_3 (Dense) Param=XXXXXX >> 是Output_Shape*128+128而來
第七層的dense_4 (Dense) Param=1290 >> 是10*128+10而來

**CNN參數的影響**(filters / keranel_size / strides / padding / pool_size)**
#詳細解說可參考py3-7 00:25:00-00:35:00
(A)activation 可選不同的mode >> 較有名的為下列
    "relu"(用CNN較優)
    "LeakyRelu"(用CNN較優)  
    "Sigmoid"(用在RNN的)
    "tanh"(用在RNN居多)
(B)filters:幫你產生幾張圖
(C)keranel_size(the size of the filter):讀一張圖時>>一次讀圖的多少區域大小
(D)strides:讀一張圖時>>一次跳多少格圖
    ###動畫詮釋strides不同數值代表甚麼意思
    strides=(1,1)>>https://www.cntk.ai/jup/cntk103d_same_padding_no_strides.gif
    strides=(2,2)>>https://www.cntk.ai/jup/cntk103d_padding_strides.gif
(E)padding:same>>有包含圖的邊框 / 'valid' (不包含圖像邊框,參數會就少,但可能影響準確率)
(F)pool_size >> 縮圖的概念>>也就是每2*2縮成一格(MaxPooling2D>>縮成的一格值,是取2*2裡面最大值)

**英文表達**
(A)the number of channels in the input image is 3 >> (28,28,3)
(B)the width and height of the input image is (28,28) >> 圖為(28,28)
(C)the size of the filter specified for the first convolutional layer is (5,5) >> kernel_size=(5, 5)
(D)the stride specified for the first convolutional layer is (1,1) >> strides=(1,1)
(E)the padding specified for the first convolutional layer is same >> padding='same'

**CNN參數參考(py3-8 00:20:00-00:45:00)**
下面是"是否用pooling layers"
     "甚麼pooling layers(MAX/AVG)"
     "甚麼activation"
     的參數>>所得到的ERROR%(越小越好)
# without maxpooling--> Max    --> AVG
# 1.37% relu        --> 1.13%  --> 1.10%
# 1.31% LeakyRelu   --> 1.23%  --> 1.16%
# 2.69% Sigmoid     --> 1.49%  --> 3.89%
# 1.40% tanh        --> 0.85%  --> 1.47%

'''

#建立模組
import keras
from keras import optimizers
model.compile(optimizer=optimizers.SGD(0.2) ,#用SGD優化模組learn rate=0.2
              loss=keras.losses.categorical_crossentropy , 
              metrics=['accuracy'])

#帶入train Data
history = model.fit(X_train, 
                    y_train_oh, 
                    epochs= 5, #epoch所代表的數字是指所有數據被訓練的總輪數
                    verbose=1, #0>>不顯示訊息 / 1>>immediate顯示訊息 / 2>>buffer顯示訊息
                    validation_split=0.2#validation_split>>用剛剛的訓練資料去驗證  / validation_data>>另外再給驗證資料
                    )

#用模組驗證資料
Train_loss , Train_acc = model.evaluate(X_train, y_train_oh)
print('Train_Loss值: ',Train_loss,'Train準確率: ',Train_acc)
Test_loss , Test_acc = model.evaluate(X_test, y_test_oh)
print('Test_Loss值: ',Test_loss,'Test準確率: ',Test_acc)

#將準確度/loss值列出來
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
print('acc:',acc,' val_acc:',val_acc,' loss:',loss,' val_loss:',val_loss)
#***loss值 不等於 錯誤率(1-準確率)***
#***loss值 不等於 錯誤率(1-準確率)***
#***loss值 不等於 錯誤率(1-準確率)***


#將Train與Test的acc值畫圖
import matplotlib.pyplot as plt
#history.epoch>>epoch了幾次
plt.plot(range(len(history.epoch)), history.history['acc'], color='red'  )
plt.plot(range(len(history.epoch)), history.history['val_acc'], color='blue'  )
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.grid()
plt.show()

#將Train與Test的loss值畫圖
import matplotlib.pyplot as plt
plt.plot(range(len(history.epoch)), history.history['loss'], color='red'  )
plt.plot(range(len(history.epoch)), history.history['val_loss'], color='blue'  )
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()