# 高光谱分类

## 1.主要思路

1.先利用os批量提取文件路径名，再用scipy读取.mat数据，再用panda 转化成csv。且添加进去样本标签

2.将各类样本合并成一个，并且分成train 和 lable

3.样本中心化

4使用svc确定参数训练模型

## 2.代码部分及成果

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
import pandas as pd
from pandas import Series,DataFrame
import pandas as pd
import scipy
from scipy import io
import os
import csv
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
```



```python
#第一步数据处理，合并，格式转化。
##1.Data batch processing
file_dir = r'/jupyternotebook/train_nine/'#创建三个列表，分别存储目标目录，mat格式中数据名，输出目录
pathlist = []
namelist = []
resultlist = []
for root,dirs,files in os.walk(file_dir):#批量提取文件路径
    for file in files:
        if os.path.splitext(file)[1] == '.mat':
            pathlist.append(os.path.join(root,file))
            namelist.append(os.path.splitext(file)[0])
        if  os.path.splitext(file)[1] == '.csv':
            resultlist.append(os.path.join(root,file))
for i in range(0, 9):#将mat格式转化为csv且加上lable
    a = pathlist[i]
    b = namelist[i]
    features_struct = scipy.io.loadmat(a)
    features = features_struct[b]
    dfdata = pd.DataFrame(features)
    c = [10,11,12,14,2,3,5,6,8]
    dfdata['200']= c[i]
    datapath1 = resultlist[i]
    dfdata.to_csv(datapath1, index=False)
#读取测试集
d = pathlist[9]
test = scipy.io.loadmat(d)
ts = test[namelist[9]]
tsf = pd.DataFrame(ts)
tspath = resultlist[9]
tsf.to_csv(tspath,index=False)
for i in range(0,len(pathlist)-1):#合并各类数据
    df = pd.read_csv(resultlist[i])
    df.to_csv(file_dir+'/'+ 'allset.csv',encoding="utf_8_sig",index=False,  mode='a+')
##Data first split
allset = pd.read_csv('/jupyternotebook/train_nine/allset.csv')#分割成train__data,和train_label
train_label = allset['200']
train_data = allset.drop(['200'],axis=1)
train_data.to_csv(file_dir+'/'+ 'train_data.csv',header=False)
train_label.to_csv(file_dir+'/'+ 'train_label.csv',header=False)



```







```python
##训练模型
from sklearn import  preprocessing#数据中心化
train_data = preprocessing.scale(train_data)
data_train, data_test, label_train, label_test = train_test_split(train_data,train_label,test_size=0.3)#切割测试集训练集
##构建模型 训练
clf = SVC(kernel='rbf',C=5)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print (accuracy)


```







```python
ts = preprocessing.scale(ts)#运行测试机集
final_pred = clf.predict(ts)
print(final_pred)
fpd = pd.DataFrame(final_pred)
fpd.to_csv('/jupyternotebook/train_nine/jieguo.csv')

```

