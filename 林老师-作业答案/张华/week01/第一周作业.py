#导入numpy包
import numpy as np 

#创建一个(3,4)的二维矩阵
a1 = np.array([(1,2,3,3), (4,5,6,6), (7,8,9,10)])
print(a1)
print(a1.dtype)#如果没有显式指定 dtype，NumPy 会根据输入数据自动推断一个合适的类型

#创建一个(2,3)的二维矩阵，矩阵里面的元素全都是0
a2=np.zeros((2,3), dtype=np.float32)
print(a2)
print(a2.dtype)
a2=np.ones((3,3),dtype=np.int16)
print(a2)

#创建一个在1到6之前的一维等差矩阵，步长为0.4
a3=np.arange(1,6, 0.4)
print(a3)

#创建一个(4,4)的单位矩阵（主对角线上是1，其他地方全都是0）
a4=np.eye(4)
print(a4)

#创建一个6个随机数（元素取值0-1）的一维矩阵
a5 = np.random.random(6)
print(f'a5 : {a5}')

#创建一个均值为0，标准差为0.1，拥有5个样本的一维矩阵
a6=np.random.normal(0, 0.1, 5)
print(f'a6 : {a6}')

#切片操作 【start end step】
#包含start，不包含end
a7 = np.array([(1,2), (3,4), (5,6)])
print(a7[:,1]) #第一维全取，第二维取第1列


a8 = np.array([(1,2), (3,4), (5,6)])
# i,j = a8[0]
for i,j in a8:
    print(i,j)

a = np.array([(1,2,3,0), (4,5,6,9), (7,4,8,9)])
'''
ndim：数组的维度数
shape：数组的形状（每个维度的大小）
size：数组的总元素个数
dtype：数组元素的数据类型
'''
print("ndim:", a.ndim)
print("shape:", a.shape)
print("size", a.size)
print("dtype", a.dtype)
print(3 in a) #3是否在矩阵中

a7 = np.arange(1,19)
print(a7)
print(a7.shape)

a7 = a7.reshape(3,3,2)  # 维度大小乘积 == 元素个数 重新排列
print(a7)

#转置
print(a)
a = a.T
print(a)

#平铺为一维矩阵
a = a.flatten()
print(a)

#插入新维度
a8 = np.array([(1,2,7), (3,4,8), (5,6,0)]) #(3,3)
print(a8.shape)
a8 = a8[:,:,np.newaxis]  # [3,3,1]
print(a8.shape)
print(a8)

#矩阵运算
a = np.ones((3,3))
b = np.array([(-1,1,1),(-1,1,-1),(-1,1,-1)])
print(a+b)
print(a-b)
print(a.sum()) #元素和
print(a.prod()) #元素相乘

a = np.array([50,3,11])
print("mean:",a.mean())#均值（mean）
print("var:", a.var())#方差（var）
print("std:", a.std())#标准差（std）

a = np.array([1.02, 3.8, 9])
print("argmax:", a.argmax()) # 最大值的索引
print("argmin:", a.argmin()) # 最小值的索引

print("ceil:", np.ceil(a))  # 向上取整
print("floor:", np.floor(a)) # 向下取整
print("rint:", np.rint(a))  # 四舍五入

a = np.array([16,31,12,28,22,31,48])
a.sort()  # 排序
print(a)

#导入 pytorch包
import torch

#创建张量
data = torch.tensor([[1,2,6],[3,4,9]], dtype=torch.float32) # 定义一个2*3的矩阵 
print(data)

import numpy as np

#创建张量
np_array = np.array([[1,2,6],[3,4,9]])
data2 = torch.from_numpy(np_array)
print(data2)

# 通过已知张量维度，创建新张量
data3 = torch.rand_like(data2, dtype=torch.float)
print(data3)

shape = (2,3,8)
rand_tensor = torch.rand(shape)# 随机初始化
ones_tensor = torch.ones(shape)# 全1初始化
zeros_tensor = torch.zeros(shape)# 全0初始化

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 基于现有tensor构建，但使用新值填充
m = torch.ones(3,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)

# 获取tensor的大小
print(m.size()) # torch.Size([5,3])

# 均匀分布
# 每个元素是从 [0, 1) 之间的均匀分布中随机采样
print(torch.rand(3,3))
# 标准正态分布
# 生成的值是从标准正态分布（均值为 0，标准差为 1）中采样的
print(torch.randn(3,3))
# 离散正态分布
# 生成的值是从标准正态分布（均值为 0，标准差为 1）中采样的
print(torch.normal(mean=.0,std=1.0,size=(5,3)))
# 线性间隔向量(返回一个1维张量，包含在区间start和end上均匀间隔的steps个点)
print(torch.linspace(start=1,end=10,steps=21))

# 张量属性
tensor = torch.rand(3,5)
print(f"value of tensor: {tensor}")
# 张量的形状、数据类型和存储设备
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# 检查pytorch是否支持GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

# mac上没有GPU，使用M系列芯片
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = tensor.to(device)

print(tensor)
print(tensor.device)

# 张量的切片操作
tensor = torch.tensor([[1,2,3],[4,5,6]])
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0 # 把第二列的值全部变为0
print(tensor)

# 张量的拼接操作
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1 * 3) # 张量的乘法
print(t1.shape)


import torch
tensor = torch.arange(1,17, dtype=torch.float32).reshape(4, 4)

# 计算两个张量之间矩阵乘法的几种方式。 y1, y2, y3 最后的值是一样的 dot
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

print("y1",y1)
print("y2",y2)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print("y3",y3)


# 计算张量逐元素相乘的几种方法。 z1, z2, z3 最后的值是一样的。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
print(z3)

agg = tensor.sum() # 求和
print(agg) #打印的是张量
agg_item = agg.item() # 取pytorch中的值，转为python中的值
print(agg_item, type(agg_item))

np_arr = z1.numpy()# 张量转numpy
print(np_arr)

print(tensor, "\n")
# 原地操作会直接修改原张量，因此使用时要小心，避免意外修改数据。
tensor.add_(5)# 原地操作
# tensor = tensor + 5 不会修改原张量
# tensor += 5 不会修改原张量
print(tensor)