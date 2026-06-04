import torch

# x = torch.tensor(3.0, requires_grad=True)

# y = x ** 2

# y.backward()

# print(x.grad)

# x = torch.tensor(2.0, requires_grad=True)

# z = x + 3
# y = z ** 2
# y.backward()
# print(x.grad)

# x = torch.randn(3, requires_grad=True)   # 三个输入
# y = (x ** 2).sum()                       # 先平方，再求和 → 标量
# y.backward()
# print(x.grad)                            # 梯度是 dy/dx_i = 2*x_i

# # 假设有模型参数 w 和 b
# w = torch.randn(1, requires_grad=True)
# b = torch.randn(1, requires_grad=True)

# # 训练数据
# x = torch.tensor([2.0])
# y_true = torch.tensor([5.0])

# # 前向传播：计算预测值
# y_pred = w * x + b

# # 计算损失（标量）
# loss = (y_pred - y_true) ** 2

# # 自动求导：计算 loss 对 w 和 b 的梯度
# loss.backward()

# # 此时 w.grad 和 b.grad 已经填好了
# print(w.grad)   # 梯度值
# print(b.grad)

# # 用优化器（如 SGD）更新参数
# with torch.no_grad():
#     w -= 0.1 * w.grad
#     b -= 0.1 * b.grad

# # 清零梯度，为下一次迭代做准备
# w.grad.zero_()
# b.grad.zero_()

# x = torch.tensor(2.0, requires_grad=True)
# y = 3 * x ** 2 + 2 * x + 1
# y.backward()
# print(x.grad)
a = torch.tensor([1., 2., 3.], requires_grad=True)
b = torch.tensor([4., 5., 6.])
dot = (a * b).sum()
dot.backward()
print(a.grad)

# 练习3：观察梯度累积
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)   # 2.0
y = x ** 3
y.backward()    # 注意：没有清零，梯度会累积
print(x.grad)   # 2.0 + 3.0 = 5.0
x.grad.zero_()  # 手动清零
print(x.grad)   # 0.0