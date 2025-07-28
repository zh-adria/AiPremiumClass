### PyTorch逻辑回归知识点总结笔记

#### **一、逻辑回归核心概念**

1. **与线性回归的区别**

   - **线性回归**：拟合连续值，直接输出实数。
   - **逻辑回归**：通过Sigmoid函数将线性输出映射为概率（0~1），解决二分类问题。
   - 决策边界：*z*=*w**T**x*+*b*=0，当概率 > 0.5 时判为正类，否则为负类。

2. **Sigmoid函数**
   ![image-20250306184521888](C:\Users\yuxiangw\AppData\Roaming\Typora\typora-user-images\image-20250306184521888.png)

   - 将线性输出压缩到 (0,1) 区间，表示概率。

   - 代码实现：

     ```Python
     def sigmoid(z):
         return 1 / (1 + np.exp(-z))
     ```

#### **二、数学基础**

1. **最大似然估计（MLE）**

   - 目标：找到参数 θ*θ*，使观测数据出现的概率最大。
   - ![image-20250306184502171](C:\Users\yuxiangw\AppData\Roaming\Typora\typora-user-images\image-20250306184502171.png)

2. **交叉熵损失函数**

   - 负对数似然函数：
     ![image-20250306184710366](C:\Users\yuxiangw\AppData\Roaming\Typora\typora-user-images\image-20250306184710366.png)

   - 代码实现（防溢出）：

     ```Python
     def loss_function(y, y_hat):
         epsilon = 1e-8  # 防止 log(0)
         return - (y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
     ```

3. **梯度计算**

   - ![image-20250306184334002](C:\Users\yuxiangw\AppData\Roaming\Typora\typora-user-images\image-20250306184334002.png)

#### **三、训练流程**

1. **数据准备**

   - 标准化特征（Z-Score）加速收敛。

   - 划分训练集/测试集（比例7:3），防止数据泄露。

   - 代码示例：

     ```Python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
     ```

2. **参数初始化**

   - 权重 w*w*：从正态分布 N(0,0.01)N(0,0.01) 初始化。

   - 偏置 b*b*：初始化为0。

   - PyTorch实现：

     ```Python
     w = torch.randn(1, n_features, requires_grad=True)
     b = torch.zeros(1, requires_grad=True)
     ```

3. **前向传播**

   - 线性部分：z*=*w**T**x*+*b

   - Sigmoid激活：y^=σ(z)

   - PyTorch代码：

     ```Python
     z = torch.mm(X, w.t()) + b
     y_hat = torch.sigmoid(z)
     ```

4. **损失计算与反向传播**

   - 计算损失：

     ```Python
     loss = F.binary_cross_entropy(y_hat, y)
     ```

   - 反向传播：

     ```Python
     loss.backward()  # 自动计算梯度
     ```

5. **参数更新**

   - 梯度下降公式：
     ![image-20250306184808074](C:\Users\yuxiangw\AppData\Roaming\Typora\typora-user-images\image-20250306184808074.png)

   - PyTorch实现：

     ```Python
     with torch.no_grad():
         w -= learning_rate * w.grad
         b -= learning_rate * b.grad
         w.grad.zero_()  # 梯度清零
         b.grad.zero_()
     ```

#### **四、PyTorch实现细节**

1. **自动求导机制**

   - 使用 `requires_grad=True` 跟踪计算图。
   - `loss.backward()` 自动计算梯度。
   - 在参数更新时关闭梯度跟踪：`with torch.no_grad()`。

2. **完整训练代码示例**

   ```Python
   import torch
   from sklearn.datasets import make_classification
   
   # 数据准备
   X, y = make_classification(n_features=10)
   X = torch.tensor(X, dtype=torch.float32)
   y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
   
   # 参数初始化
   w = torch.randn(1, 10, requires_grad=True)
   b = torch.zeros(1, requires_grad=True)
   
   # 超参数
   learning_rate = 0.01
   epochs = 1000
   
   # 训练循环
   for epoch in range(epochs):
       # 前向传播
       z = torch.mm(X, w.t()) + b
       y_hat = torch.sigmoid(z)
       
       # 计算损失
       loss = F.binary_cross_entropy(y_hat, y)
       
       # 反向传播
       loss.backward()
       
       # 参数更新
       with torch.no_grad():
           w -= learning_rate * w.grad
           b -= learning_rate * b.grad
           w.grad.zero_()
           b.grad.zero_()
       
       if epoch % 100 == 0:
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
   ```

#### **五、超参数调优**

1. **学习率（η）**
   - **过大**：梯度震荡，难以收敛。
   - **过小**：收敛速度慢。
   - 建议初始值：0.01，通过网格搜索调整。
2. **梯度问题处理**
   - **梯度爆炸**：梯度裁剪 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`。
   - **梯度消失**：使用Xavier初始化或Batch Normalization。