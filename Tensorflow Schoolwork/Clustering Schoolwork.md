# Tensorflow Schoolwork

## 1.整体思路

首先我用的是tensorflow搭建的模型包括一下几步

1.iris数据集分割

2.构建k近邻模型

3.训练模型

4.正确率的计算

## 2.成果展示

#### 1.分割及构建

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split
# 开始设计模型
#分割数据集
x_train, x_test, y_train, y_test = train_test_split(
 iris.data, iris.target, test_size=0.2)
# 输入占位符
xtr = tf.placeholder("float", [None, 4])
xte = tf.placeholder("float", [4])
# 计算L1距离
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 获取最小距离的索引
pred = tf.arg_min(distance, 0)

#分类精确度
accuracy = 0.

# 初始化变量
init = tf.global_variables_initializer()
```

![](C:\Users\lenovo\Desktop\github\18b-zhangchong-2016-331\Tensorflow Schoolwork\image\splitresult.png)

 					分割成果

#### 2.开始训练并且计算争取率

```python
# 运行会话，训练模型
with tf.Session() as sess:

    # 运行初始化
    sess.run(init)

    # 遍历测试数据
    for i in range(len(x_test)):
        # 获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={xtr:x_train, xte:x_test[i,:]})   #向占位符传入训练数据
        # 最近邻分类标签与真实标签比较
        print("Test", i, "Prediction:", (y_train[nn_index]), \
        # 计算精确度 验证模型
        if y_train[nn_index] == y_test[i]:
            accuracy += 1./len(x_test)

    print("Done!")
    print("Accuracy:", accuracy)

```

![](C:\Users\lenovo\Desktop\github\18b-zhangchong-2016-331\Tensorflow Schoolwork\image\finalresult.png)

​				最终训练结果正确率



## 3.细节描述

1.分割数据集使用的是sklearn里的train_test_split。在查了文档后掌握使用了

2.争取率的计算及通过对比预测样本和实际样本标签是否一致得出

## 张冲 2016011331