import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import datetime

df1 = pd.read_csv('new_data.csv')
df2 = pd.read_csv('test_set.csv')
print(df1.head())
print(df2.head())

'''
使用 Scikit-Learn 库来进行培训/测试拆分。
之后，我们将把拆分后的数据从 Numpy 数组转换为 PyTorch tensors。
首先，我们需要将Iris数据集拆分为特征和目标--或者说是X和y，列Name将是目标变量，其他的都是特征（或预测因子）。
我还将使用随机种子，所以你能够重现我的结果。
'''
X_train = df1.drop(['gameId', 'creationTime', 'winner'], axis=1).values
X_test = df2.drop(['gameId', 'creationTime', 'winner'], axis=1).values
print(X_train)
y_train = df1['winner'].values
y_test = df2['winner'].values
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 前几行是这样的。
print(X_train[0:3, ])
print(y_train[0:3, ])

'''
我们将构建一个3层MLP：
1.全连接隐藏层（18个输入特征（X中的特征数量），8个输出特征（任意））
2.输出层（8个输入特征（上一层的输出特征数量），3个输出特征（不同类的数量））
这是最简单的MLP，只包括1个隐藏层。
除此之外，我们将使用Sigmoid作为我们的激活函数。
'''
class ANN(nn.Module):
    # 在构造函数中，你将定义所有的层及其架构
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=18, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=3)

    # 在forward()方法中，你将定义一个前传
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.output(x)
        x = F.softmax(x)
        return x


'''
PyTorch 使用这种面向对象的方式来声明模型，而且相当直观。
现在让我们制作一个模型的实例，并验证它的架构是否与我们上面指定的架构一致。
'''
model = ANN()

'''
在训练模型之前，我们还需要声明一些东西。
准则: 基本上我们如何衡量损失，我们将使用CrossEntropyLoss。
优化器：优化算法，我们会使用学习率为0.01的Adam，下面是如何用代码实现。
'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

'''
模型训练
我们将对模型进行100个纪元的训练，跟踪时间和损失
每隔2个纪元，我们将向控制台输出当前的状态--表明我们在哪个纪元，当前的损失是多少
'''
starttime = datetime.datetime.now()
epochs = 100
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X_train)
    loss = criterion(y_hat, y_train)
    loss_arr.append(loss)
    if i % 2 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

endtime = datetime.datetime.now()
print('The training time is', (endtime - starttime).microseconds/1000,'ms')

# 训练结束后，我们在测试集中应用模型。之后，我们仍然应用softmax函数来得到分类结果。
predict_out = model(X_test)
_, predict_y = torch.max(predict_out, 1)
print(predict_y)

# 我们可以利用Sklearn中的库进一步比较它的准确度得分。
from sklearn.metrics import accuracy_score
print("The accuracy is ", accuracy_score(y_test, predict_y))
