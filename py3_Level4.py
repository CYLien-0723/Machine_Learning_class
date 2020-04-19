# -*- coding: utf-8 -*-
"""
(一)要用deep learning and 用GPU跑的時候 要裝下面這些東西
    --Visual Studio(筆電已經裝了2015的)
    --CUDA(版本要10才能匹配cntk==2.6) + cuDNN() 
    --tensorflow-gpu==1.10.0 + tensorflow
    --cntk-gpu==2.6 + cntk==2.6
    --python3.6.x
    
(二)如果沒有GPU則只要裝下面的東西即可(差別只在速度)
    --tensorflow
    --cntk==2.6
    --python3.6.x
    
(三)簡單辨認有沒有用到GPU方式
    from cntk import device
    for dev in device.all_devices():
        print(dev)
    
(四)下面老師將會帶我們做巨匠edx(openedx.pcschool.com.tw)的DAT236x的課
    基本上裡面的課分好幾章節,每個章節都有一個ipynb(程式+筆記)
    按照上面的程式去解答每一章節的題目來完成這個線上課程
    py3_Level4 會跟 py3_Level2很像(或說根本在做一樣的事)
    py3_Level2 是再用keras做(很簡單省事)
    py3_Level4 是用cntk作(相對起來挺複雜的)
    主要是老師在帶我們練習做DAT263x的課程題目
    
(五)一些np用法
    np.eye用法如下:
        np.eye(5)=
        array([[1,0,0,0,0]
               [0,1,0,0,0]
               [0,0,1,0,0]
               [0,0,0,1,0]
               [0,0,0,0,1]])

"""


#======================DAT236x(第一章)========================
#主要為下載data(下面為ipynb的code)
from __future__ import print_function
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os,shutil,struct,sys,gzip
try: from urllib.request import urlretrieve 
except ImportError: from urllib import urlretrieve
def loadData(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))
def loadLabels(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:os.remove(gzfname)
    return res.reshape((cimg, 1))
def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

#------下載資料庫------(分Train(60000, 785) 與 test(10000, 785) )
url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
num_train_samples = 60000
print("Downloading train data")
train = try_download(url_train_image, url_train_labels, num_train_samples)

url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
num_test_samples = 10000
print("Downloading test data")
test = try_download(url_test_image, url_test_labels, num_test_samples)  
print(train.shape,test.shape)

#------隨便輸出一個資料的圖來看看------
sample_number = 5441
plt.imshow(train[sample_number,:-1].reshape(28,28), cmap="gray_r")
plt.axis('off')
print("Image Label: ", train[sample_number,-1])

#------將Train 與 Test data存成指定的資料模式
#將每筆資料存成( |labels {10個數字}|features {784個數字}\n )>>CNTK的資料模式
def savetxt(filename, ndarray):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.isfile(filename):
        print("Saving", filename )
        with open(filename, 'w') as f:
            labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
            for row in ndarray:
                row_str = row.astype(str)
                label_str = labels[row[-1]]
                feature_str = ' '.join(row_str[:-1])
                f.write('|labels {} |features {}\n'.format(label_str, feature_str))
    else:print("File already exists", filename)
data_dir = "C:/python3_class/Level4/data/MNIST"#自行指定要放的地方
print ('Writing train text file...')
savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)
print ('Writing test text file...')
savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)
print('Done')

#======================DAT236x(第二章)========================
#製作機器學習模組(下面我會穿插keras>>來了解這部分的cntk再幹嗎?)
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys,os
import cntk as C

#下面主要是驗證你有沒有GPU
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu': C.device.try_set_default_device(C.device.cpu())
    else: C.device.try_set_default_device(C.device.gpu(0))

#下面主要是說她後面用的CNTK的版本為2.0
if not C.__version__ == "2.0":
    raise Exception("this lab is designed to work with 2.0. Current Version: " + C.__version__) 
    
#主要是設定CNTK的隨機因子
np.random.seed(0)
C.cntk_py.set_fixed_random_seed(1)
C.cntk_py.force_deterministic_algorithms()


#下面主要是讀取第一章所存取的檔案(它是用一段一段讀以避免檔案過大~幾百G之類的)
def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    deserailizer = C.io.CTFDeserializer(path, C.io.StreamDefs(labels = labelStream, features = featureStream))
    return C.io.MinibatchSource(deserailizer,
       randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)

#確定有要讀的檔案存在
data_found = False
data_dir = "C:/python3_class/Level4/data/MNIST"#自行指定要放的地方
train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
if os.path.isfile(train_file) and os.path.isfile(test_file):data_found = True
if not data_found: raise ValueError("Please generate the data by completing Lab1_MNIST_DataLoader")
print("Data directory is {0}".format(data_dir))

#將模組的神經層建立
input_dim = 784
num_output_classes = 10
def create_model(features):
    with C.layers.default_options(init = C.glorot_uniform()):#這行主要是做"初始化"(keras也有>>kernel_initializer>>但py3_Level2沒用)
        r = C.layers.Dense(num_output_classes, activation = None)(features)#這行是建立神經層
        return r
'''
上面的code等同下面
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(128, input_shape=(784,),activation='relu'))
model.add(Dense(10, activation='softmax'))
'''



#將資料標準化~~並建立模組
input_s = input/255
z = create_model(input_s)#將模組建立在z裡面
label_error = C.classification_error(z, label)
learning_rate = 0.2#學習率
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
learner = C.sgd(z.parameters, lr_schedule)#
trainer = C.Trainer(z, (loss, label_error), [learner])

#自己製作顯示進度的方式
def moving_average(a, w=5):#每5個算一個次進度
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
    return mb, training_loss, eval_error

#開始將資料餵進去到模組裡面
minibatch_size = 64#同batch_size
num_samples_per_sweep = 60000
num_sweeps_to_train_with = 10#同epochs
num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
reader_train = create_reader(train_file, True, input_dim, num_output_classes)
input_map = {
    label  : reader_train.streams.labels,
    input  : reader_train.streams.features
} 
training_progress_output_freq = 500
plotdata = {"batchsize":[], "loss":[], "error":[]}
for i in range(0, int(num_minibatches_to_train)):
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)#如同keras的 keras.losses.categorical_crossentropy>> #偏差值function的設定
        plotdata["error"].append(error)#答案與預測答案的差
'''
上面的code等同下面
將x_train裡面的數值從0~255變成0~1
x_train_32 = x_train.astype('float32')/255.
x_test_32 = x_test.astype('float32')/255.

將x_train裡面的資料從(60000,28,28)轉成(60000,784)
x_train_32 = x_train_32.reshape(60000,784)
x_test_32 = x_test_32.reshape(10000,784)

將y_train裡面的資料轉成one-hot encoding
import keras
y_train_oh= keras.utils.to_categorical(y_train, 10)#10>>因為我們數值種類只有10種
y_test_oh= keras.utils.to_categorical(y_test, 10)#10>>因為我們數值種類只有10種

建立模型
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(128, input_shape=(784,),activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, #偏差值function的設定
              optimizer=keras.optimizers.SGD(lr=0.2), #優化器(運算的模型)>>用甚麼優化器就要記得import / lr>學習率
              metrics=['accuracy'])
'''

#畫圖
plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
plt.show()

'''
上面的code等同下面
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
'''


#讀取test資料
reader_test = create_reader(test_file, False, input_dim, num_output_classes)
test_input_map = {
    label  : reader_test.streams.labels,
    input  : reader_test.streams.features,
}
test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples // test_minibatch_size
test_result = 0.0
for i in range(num_minibatches_to_test):
    data = reader_test.next_minibatch(test_minibatch_size,input_map = test_input_map)
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error
print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))

'''
上面的code等同下面
score = model.evaluate(x_test_32 , y_test_oh,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
print('Average test error:',(1-scord[1])*100,'%')
'''

#列出前25筆答案與模組預測出來的答案
out = C.softmax(z)
reader_eval = create_reader(test_file, False, input_dim, num_output_classes)
eval_minibatch_size = 25
eval_input_map = {input: reader_eval.streams.features} 
data = reader_test.next_minibatch(eval_minibatch_size, input_map = test_input_map)
img_label = data[label].asarray()
img_data = data[input].asarray()
predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]
pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]
print("Label    :", gtlabel[:25])
print("Predicted:", pred)
'''
上面的code等同下面
print(y_test[:25])
print(model.predict_classes(x_test_32[:25]))
'''

#列出第5個樣本實際答案的圖 與 模組預測的答案
sample_number = 5
plt.imshow(img_data[sample_number].reshape(28,28), cmap="gray_r")#實際答案的圖
plt.axis('off')
img_gt, img_pred = gtlabel[sample_number], pred[sample_number]
print("Image Label: ", img_pred)#預測的答案
