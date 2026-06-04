import torch

a = torch.tensor([1, 2, 3])
a = torch.tensor([[1, 2], [3, 4]])
a = torch.zeros(2, 3)
a = torch.ones(3, 4)
a = torch.rand(3, 2)

import numpy as np

arr = np.array([5, 6, 7])
# print(type(arr))
c = torch.from_numpy(arr)
# print(type(c))
back_to_numpy = c.numpy()
# print(type(back_to_numpy))
# print(a)

x = torch.rand(2, 3, 4)
# print(x.shape)
# print(x.dtype)
# print(x.device)

# x = torch.tensor([[10, 20, 30],
#                   [40, 50, 60]])

# print(x[0, 1])    # 第0行第1列 → 20
# print(x[:, 1])    # 所有行的第1列 → [20, 50]
# print(x[0, :])    # 第0行所有列 → [10, 20, 30]

# x = torch.arange(7)
# print(x)

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 逐元素运算
# print(a + b)      # [5,7,9]
# print(a * b)      # [4,10,18]  （不是矩阵乘法！）
# print(a ** 2)     # [1,4,9]

# 矩阵乘法（重要）
A = torch.rand(2, 3)
B = torch.rand(3, 4)
C = torch.mm(A, B)   # 结果形状 (2,4)
# 或者用 @ 符号
C = A @ B
# print(C)

x = torch.rand(2, 3)
# print(x)
# # 求和：可以指定对哪一维求和
# print(x.sum())           # 所有元素加起来 → 标量
# print(x.sum(dim=0))      # 把第0维（行）消掉 → 形状 (3,)
# print(x.sum(dim=1))      # 把第1维（列）消掉 → 形状 (2,)

# 拼接
a = torch.rand(2, 3)
b = torch.rand(2, 4)
c = torch.cat([a, b], dim=1)   # 横向拼接 → 形状 (2, 7)
# print(a)
# print(b)
# print(c)

# 增加/删除维度
x = torch.rand(3, 4)
y = x.unsqueeze(0)       # 在第0维前面加一维 → 形状 (1,3,4)
z = y.squeeze(0)         # 去掉第0维（如果那维大小是1）→ 回到 (3,4)
# print(x)
# print(y)
# print(z)

# practice
a = torch.rand(3, 4)
print(a)
a = a.view(2, -1)
print(a)
sum = a.sum(dim=1).unsqueeze(0)
print(sum)
trans = sum.transpose(0, 1)
print(trans)
trans *= 2
print(trans)