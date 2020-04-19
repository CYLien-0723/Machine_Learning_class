# -*- coding: utf-8 -*-
"""
Python3 新課程
第一節主要在講
1)註冊Eda(https://courses.edx.org) 與 MPP-Capstone(https://www.datasciencecapstone.org) 的帳號
2)如何從MPP-Capstone開始練習DAT264x
3)

"""

#======================================練習DAT264x========================
#-----------老師用datasets.load_digits()幫大家複習基本的東西

#(a)將資料庫拿進來
import matplotlib.pyplot as plt
from sklearn import datasets
mnist = datasets.load_digits()
mnist.keys()
plt.figure(figsize=(2,2))
plt.axis('off')
plt.imshow(mnist.data[0].reshape(8,8),cmap='gray_r')
#(b)將資料分成train 與 test data
from sklearn.model_selection import train_test_split
X = mnist.data
y = mnist.target
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
print(X.shape,y.shape)
#其中X一行就是一個8*8的圖,只是他把它展成64欄位

#(c)訓練模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='newton-cg',multi_class='multinomial')
#solver 演算法(優化器)
clf.fit(x_train,y_train)

#(d)結果
import numpy as np
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
np.sum(clf.predict(x_test))


#-----------正式開始用練習DAT264x
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#(a-1)讀取x_train
X = []#np.array([])
for ppap in range(10000,14500):
    dd = np.array([])
    for pp in [aa for aa in io.imread('C:/python/DAT264x/train/'+str(ppap)+'.png', as_gray=True)]:
        dd = np.append(dd,pp)
    X.append(list(dd))
#plt.imshow(my_image)
X = pd.DataFrame(X)
#(a-2)讀取y_train
y_df = pd.read_csv('C:/python/DAT264x/train_labels.csv')
y_df.head(10)#列出前10筆
Y = y_df.accent.values
#(b-1)將資料分成train 與 test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1,stratify=Y)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
#(b-2)先將真正要測試的資料讀近來
#讀取X_t>>真正要測試的資料
X_t = []#np.array([])
for ppap in range(20000,25377):
    dd = np.array([])
    for pp in [aa for aa in io.imread('C:/python/DAT264x/test/'+str(ppap)+'.png', as_gray=True)]:
        dd = np.append(dd,pp)
    X_t.append(list(dd))
X_t = pd.DataFrame(X_t)

#########(法一)#########>>準確率為0.53
#(c)訓練模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='newton-cg',multi_class='multinomial')#solver 演算法(優化器)
clf.fit(x_train,y_train)
#(d)結果
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))
np.sum(clf.predict(x_test))
#(d-2)另一看結果方式
#Confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
np.set_printoptions(suppress=True)
sns.set() # plot formatting
mat = confusion_matrix(y_train, clf.predict(x_test)) # change it
sns.heatmap(mat, square=True, annot=True, cbar=False, fmt='g')
plt.xlabel('Predicted value')
plt.ylabel('True value') # 

#(e)預測真正需要測試的數據
y_pred_value = clf.predict(X_t)
y_df=pd.DataFrame({"file_id":[str(i) for i in range(20000,25377)],"accent":y_pred_value})
y_df.to_csv('C:/python/DAT264x//submission20190723_LogisticRegression.csv', index=False)



#########(法二)#########>>準確率為0.62
import xgboost
#老師拋的參數~~需要跑有點久
clf = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=1000,
 max_depth=5, min_child_weight=1, gamma=0,
 subsample=0.8, colsample_bytree=0.8, objective= 'multi:softmax',
 nthread=4, scale_pos_weight=1, seed=27)

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
#讀取需要測試的數據
y_pred_value = clf.predict(X_t)
y_df=pd.DataFrame({"file_id":[str(i) for i in range(20000,25377)],"accent":y_pred_value})
y_df.to_csv('C:/python/DAT264x//submission20190723_xgb_1000_5_1_08.csv', index=False)


#########(法三)#########>>準確率為0.61
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None,random_state=1)
bag = BaggingClassifier(base_estimator=clf,n_estimators=500, 
                        max_samples=1.0,max_features=1.0, 
                        bootstrap=True,bootstrap_features=False, 
                        n_jobs=1,random_state=1)

bag = bag.fit(x_train,y_train)
bag.score(x_test,y_test)

#########(法四)#########>>準確率為0.45
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
ada = AdaBoostClassifier(base_estimator=tree,n_estimators=500, learning_rate=0.1,random_state=1)

ada = ada.fit(x_train, y_train)
ada.score(x_test,y_test)



''' XGBClassifier參數介紹
booster
    gbtree 树模型做为基分类器（默认）
    gbliner 线性模型做为基分类器
silent
    silent=0时，不输出中间过程（默认）
    silent=1时，输出中间过程
nthread
    nthread=-1时，使用全部CPU进行并行运算（默认）
    nthread=1时，使用1个CPU进行运算。
scale_pos_weight
    正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10。
n_estimatores
    含义：总共迭代的次数，即决策树的个数
    调参：
early_stopping_rounds
    含义：在验证集上，当连续n次迭代，分数没有提高后，提前终止训练。
    调参：防止overfitting。
max_depth
    含义：树的深度，默认值为6，典型值3-10。
    调参：值越大，越容易过拟合；值越小，越容易欠拟合。
min_child_weight
    含义：默认值为1,。
    调参：值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）。
subsample
    含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。
    调参：防止overfitting。
colsample_bytree
    含义：训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。
    调参：防止overfitting。
learning_rate
    含义：学习率，控制每次迭代更新权重时的步长，默认0.3。
    调参：值越小，训练越慢。
    典型值为0.01-0.2。
objective 目标函数
    回归任务
        reg:linear (默认)
        reg:logistic 
    二分类
        binary:logistic     概率 
        binary：logitraw   类别
    多分类
        multi：softmax  num_class=n   返回类别
        multi：softprob   num_class=n  返回概率
    rank:pairwise 
eval_metric
    回归任务(默认rmse)
        rmse--均方根误差
        mae--平均绝对误差
    分类任务(默认error)
        auc--roc曲线下面积
        error--错误率（二分类）
        merror--错误率（多分类）
        logloss--负对数似然函数（二分类）
        mlogloss--负对数似然函数（多分类）
gamma
    惩罚项系数，指定节点分裂所需的最小损失函数下降值。
    调参：
alpha
    L1正则化系数，默认为1
lambda
    L2正则化系数，默认为1
'''






