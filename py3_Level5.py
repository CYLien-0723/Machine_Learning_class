# -*- coding: utf-8 -*-
"""
(a)下面網址主要介紹各種神經層
    http://www.asimovinstitute.org/neural-network-zoo/
    其中主要老師教的為
    Perceptron/Feed Forward / Deep /Feed Forward(最一般的神經層)
    Recurrent Neural Network(RNN)
    Deep Convolutional Network(CNN) >> kernel模組就是專門拿來用CNN
    Generative Adversarial Nerwork(GNN)
    
(b)下面youtube直接用3D的方式讓你看不同神經層的連結方式不同的樣子(Neural Network 3D Simulation)
    Perceptron [783>>10](input layer/output layer)  ---DAT236x_ch2介紹的
    ML(Multilayer) Perceptron[784>>400>>200>>10](input layer/hidden layer/output layer)  ---DAT236x_ch3介紹的
    Convolutional Neural Network(CNN) [784>>kernal>>Perceptron]  ---DAT236x_ch??介紹的
    
(c)老師淺談為何教keras 而比較不用tensorflow cntk
    (一)tensorflow 2.0 跟keras非常相似
    (二)因為DAT236x課程是教你用cntk>>但業界很草再用cntk(看py3_Level2 與 py3_Level4對比就會知道了)
    (三)cntk是微軟的框架>>然而已經很少再繼續研發>>所以資料很少
    (三)我們因該專注於神經層如何堆疊使yield變好>>所以keras是最方便的
    
(d)老師淺談 Machine Learning V.S Deep Learning
    (1)Machine Learing 
    	>>需要人為去辨別特徵(那些特徵該取那些不該)
    	>>比較適用在有結構(structtured)的資料>>所以基本上都是有建立資料庫的
    	>>ex:學生成績/購房能力/分公司設立據點...等等
    (2)Deep Learing
    	>>特徵值皆由機器取取
    	>>適用非結構的資料
    	>>ex:圖片分辨/影音/聲音/語言...等等

(e)講解DAT236x的ch2-Q6問題並且講解看cntk的code改成keras的>>請看回py3_Level2的code

(f)在下面code有一個 activation 稱為"啟動函數">>用來判斷"中間層"的節點,訊號要不要繼續傳給下一個神經層的涵式
    model = Sequential()
    model.add(Dense(128, input_shape=(784,),activation='relu'))
    model.add(Dense(10, activation='softmax'))
    最有名的函數為以下幾個:
    sigmoid / Tang / Relu / Leaky(Parametric Relu) >>這些函數長怎樣請看py3-5 1:56:14

(g)教如何將圖片load近來的方式:
    (一)from PIL import Image
        img = Image.open(image_path)#不過這個img不是一個矩陣而是一張圖

    (二)import matplotlib.pyplot as plt
        img2 = plt.imread(image_path)
        plt.imshow(img2)
        plt.show()
        
    (三)from skimage import io
        import matplotlib.pyplot as plt
        img3 = io.imread(image_path, as_gray=True)
        plt.imshow(img3, cmap='gray_r')
        plt.show()

(h)介紹一維 array如何轉換成one-hot-encoder  >> 再將one-hot-encoder轉成array
    #-------------y_train轉one-hot-encoder-----------------
    (方式一)
        label = np.unique(y_train)
        print(label)
        print(y_train[:3])
        oh_train = np.zeros([y_train.shape[0], len(label)]) #zero功能介紹在下面
        for i, v in enumerate(y_train):   # enumerate功能介紹在下面
            oh_train[i] = np.where(label == v, 1, 0)# where功能介紹在下面
    (方式二)
        from keras.utils import to_categorical
        oh_train = to_categorical(y_train)
        
    #-------------one-hot-encoder轉y_train-----------------
    y_train_return = np.argmax(oh_train,axis=1)


(i) enumerate用法如下:
    list1 = ["这", "是", "一个", "测试"]
    for i, item in enumerate(list1):
        print( i, item)
    則會印出如下:
    0 这
    1 是
    2 一个
    3 测试

(j)np.where用法如下:
    aa = np.arange(10)
    >>> np.where(aa,1,-1)
    array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
    >>> np.where(aa > 5,1,-1)
    array([-1, -1, -1, -1, -1, -1,  1,  1,  1,  1])
    
(k)np.zeros用法如下:np.zeros([行數,列數])
    >>> np.zeros([2,1])
    array([[ 0.],
           [ 0.]])
    >>> np.zeros([2,2])
    array([[ 0.,  0.],
           [ 0.,  0.]])
    >>> np.zeros([1,2])
    array([[ 0.,  0.]])    
    
(l)np.argmax用法如下:
    >>> a = np.arange(6).reshape(2,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.argmax(a),因為[0,1,2,3,4,5]>>5大,所以返回5(index)
    5
    >>> np.argmax(a, axis=0)# 0代表列,也就是[0,3]>>3大,所以返回1(index)/[1,4]返回1/[2,5]返回1
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)# 1代表行,也就是[0,1,2]>>2大,所以返回2(index)/[3,4,5]返回2
    array([2, 2])
    >>> b = array([0, 5, 2, 3, 4, 5])
    >>> np.argmax(b) # 只返回第一次出现的最大值的索引
    1

"""
#=====================下面是用 ML(Multilayer) Perceptron來跑 >>並做DAT236x的ch3-Q6====================
import matplotlib.pyplot as plt
from sklearn import datasets
mnist = datasets.load_digits()
mnist.keys()

from keras import datasets
(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape))

import numpy as np
print(np.min(x_train[0]),np.max(x_train[0]),x_train[0].dtype)
x_train_32 = np.sqrt(x_train.astype('float32')/255.)
x_test_32 = np.sqrt(x_test.astype('float32')/255.)
x_train_32 = x_train_32.reshape(60000, 784)
x_test_32 = x_test_32.reshape(10000, 784)
print(x_train_32.shape, x_test_32.shape)

import keras
y_train_oh= keras.utils.to_categorical(y_train, 10)
y_test_oh= keras.utils.to_categorical(y_test, 10)
print(y_train_oh.shape,y_test_oh.shape)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(50, input_shape=(784,),activation='sigmoid') )#這邊如果把relu改成softmax>>準確率會很慘
model.add(Dense(10, activation='softmax'))
#model.add(Dense(400, activation='None'))
#model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(0.2),
              metrics=['accuracy'])
print(model.summary())
result = model.fit(x_train_32 , y_train_oh,
          batch_size=64,
          epochs=10,
          verbose=1,
          validation_split=0.2)
#print(result.history)

score = model.evaluate(x_test_32, y_test_oh, batch_size=512 , verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Average test error:', (1 - score[1]) * 100, '%')

#下面老師教了幾種將圖片讀取近來的方式
image_path='C:/python3_class/Level6/MysteryNumberD.bmp'
#(方式一)
from PIL import Image
img = Image.open(image_path)
#不過這個img不是一個矩陣而是一張圖

#(方式二)
import matplotlib.pyplot as plt
img2 = plt.imread(image_path)
plt.imshow(img2)
plt.show()

#(方式三)
from skimage import io
import matplotlib.pyplot as plt
img3 = io.imread(image_path, as_gray=True)
plt.imshow(img3, cmap='gray_r')
plt.show()

#將讀進來的圖放進mode去做預測
img3_784 = img3.reshape(1,784)
ans = model.predict_classes(img3_784)
#結果ans為8>>但答案是5>>為甚麼呢?>>請看py3-6 00:20:00-00:30:00



#==============教 y_train轉one-hot-encoder 與 one-hot-encoder轉y_train===========
from keras import datasets
import numpy as np
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
print(y_train.shape)
#-------------y_train轉one-hot-encoder-----------------
#(方式一)
label = np.unique(y_train)
print(label)
print(y_train[:3])
oh_train = np.zeros([y_train.shape[0], len(label)]) #zero功能介紹在最上面
for i, v in enumerate(y_train):   # enumerate功能介紹在最上面
    oh_train[i] = np.where(label == v, 1, 0)# where功能介紹在最上面

#(方式二)
from keras.utils import to_categorical
oh_train = to_categorical(y_train)

#-------------one-hot-encoder轉y_train-----------------
y_train_return = np.argmax(oh_train,axis=1)
