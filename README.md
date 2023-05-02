> 数据集：[Kaggle-Heart Disease Dataset UCI](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci)

# 导包

```python
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random
import pandas as pd
from sklearn.metrics import accuracy_score
```

# 数据分析

```python
df = pd.read_csv(file_path)
```

可以看出有多个特征，然后用这些特征预测一个结果，是一个典型的分类问题。看 `target` 标签内容可知，这是一个二分类问题——我们自然想到要使用 sigmoid 函数。

# 数据处理

通过数据分析，我们可以看到数据中包含文字，但是每一个特征的内容其实就是那个几种，所以我们对每个特征进行标号。

```python
def change_df(df):
    df["sex"].replace({"Male" : 1,
                    "Female" : 0}, inplace=True)
    df["chest_pain_type"].replace({"Typical angina" : 0,
                                    "Atypical angina" : 1,
                                    "Non-anginal pain" : 2,
                                    "Asymptomatic" : 3}, inplace=True)
    df["fasting_blood_sugar"].replace({"Greater than 120 mg/ml" : 1,
                                    "Lower than 120 mg/ml" : 0}, inplace=True)
    df["rest_ecg"].replace({"Normal" : 0,
                        "ST-T wave abnormality" : 1,
                        "Left ventricular hypertrophy" : 2}, inplace=True)
    df["exercise_induced_angina"].replace({"Yes" : 1,
                                        "No" : 0}, inplace=True)
    df["slope"].replace({"Upsloping" : 0,
                        "Flat" : 1,
                        "Downsloping" : 3}, inplace=True)
    df["vessels_colored_by_flourosopy"].replace({"Zero" : 0,
                                                "One" : 1,
                                                "Two" : 2,
                                                "Three" : 3,
                                                "Four" : 4}, inplace=True)
    df["thalassemia"].replace({"Normal" : 1,
                            "Fixed Defect" : 2,
                            "Reversable Defect" : 3,
                            "No" : 0}, inplace=True)
    return df
```

在这之后，数据就很漂亮了，即根据一些数字和权重进行预测的二分类问题。

将 data 文件按照 8: 2 分为 train 和 test 两个文件。对于 train 和 test，将其前面 13 个特征放在一个变量，最后一个特征（target）放在一个变量，即构造 train_x，train_y，test_x，test_y。

# 建模

对于表格数据，最好用的神经网络其实就是 ANN 模型（MLP），它简单快速，更加契合表格的分析。

这里需要注意的是第一层 Linear 层的输入维度需要是 13，最后一层 Linear 层的输出维度需要是 2（因为这是 2 分类）。对最后一层输出的结果作 sigmoid，结果就是预测的分类。

模型如下：

```python
class Mymodel(paddle.nn.Layer):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.layer1 = Linear(13, 20)
        self.relu1 = paddle.nn.ReLU()
        self.dropout1 = paddle.nn.Dropout(0.2)
        self.layer2 = Linear(20, 25)
        self.relu2 = paddle.nn.ReLU()
        self.dropout2 = paddle.nn.Dropout(0.5)
        self.layer3 = Linear(25, 10)
        self.relu3 = paddle.nn.ReLU()
        self.output_layer = Linear(10, 2)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
```

# 参数准备

### 模型实例化

```python
model = Mymodel()
```

### 定义参数

```python
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
```

### 定义 GPU

```python
use_gpu = True
paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
```

# 训练

放在 model 种进行训练的需要时 tensor 张量形式。在转成张量之前，需要将数据先转为 numpy 形式，再利用 `paddle.to_tensor` 函数转为张量。这里需要注意，target 需要 dtype 为 int64 类型，因为在计算 loss 时，使用的是用于分类问题的 `F.cross_entropy` 函数，这需要接收两个**整数**张量。

```python
EPOCH_NUM = 500   # 设置外层循环次数
BATCH_SIZE = 32  # 设置batch大小

model.train()
for epoch_id in range(EPOCH_NUM):
    np.random.shuffle(train_data)
    mini_batches = [train_data[k:k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
    
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1]) # 获得当前批次训练数据     [10,13]
        y = np.array(mini_batch[:, -1:]) # 获得当前批次训练标签      [10,1]
        x_feature = paddle.to_tensor(x)     # [10,13]
        target = paddle.to_tensor(y, dtype='int64')    # [10,1]
        
        predicts = model(x_feature)
        
        loss = F.cross_entropy(predicts, target)    # [10,1]
        avg_loss = paddle.mean(loss)
        if iter_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
        
        avg_loss.backward()
        opt.step()
        opt.clear_grad()
```

训练完成后保存模型，方便进行预测。

```python
paddle.save(model.state_dict(), 'H_model.pdparams')
```

# 预测

先导入模型：

```python
param_dict = paddle.load('H_model.pdparams')
model.load_dict(param_dict)
```

改为预测模式在将 test_x 数据进行预测：

```python
model.eval()
predictions=[]
for i,dt in enumerate(test_x):
    y_pred=model(paddle.to_tensor(dt))
    predictions.append(y_pred.argmax().item())
```

计算准确率：

```python
accuracy = accuracy_score(test_y,predictions)
```

# 结果

**The accuracy of model is  87.8048780487805 %**

# 相关链接

数据集：[Kaggle-Heart Disease Dataset UCI](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci)

github：