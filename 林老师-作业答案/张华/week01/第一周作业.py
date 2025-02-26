import numpy as np #引入numpy包

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