# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # ------------------- 1. 生成“真实”数据 -------------------
# # 真实规律：y = 2 * x + 3 + 噪声
# torch.manual_seed(42)          # 固定随机种子，让结果可复现
# x = torch.linspace(0, 10, 100).reshape(-1, 1)   # 100个点，形状 (100,1)
# y_true = 2 * x + 3 + torch.randn(x.shape) * 0.5  # 加上一点随机噪声

# # ------------------- 2. 定义线性回归模型 -------------------
# # 一个线性层：y = w * x + b，其中 w 和 b 是需要学习的参数
# model = nn.Linear(in_features=1, out_features=1)

# # 查看模型初始参数（随机初始化）
# print(f"初始 w: {model.weight.item():.4f}, 初始 b: {model.bias.item():.4f}")

# # ------------------- 3. 定义损失函数和优化器 -------------------
# criterion = nn.MSELoss()                     # 均方误差损失
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降，学习率 0.01

# # ------------------- 4. 训练循环 -------------------
# epochs = 500
# for epoch in range(epochs):
#     # 前向传播：用当前模型预测 y
#     y_pred = model(x)                # 输入 x，输出预测值
    
#     # 计算损失
#     loss = criterion(y_pred, y_true)
    
#     # 反向传播：清空旧梯度 → 计算当前梯度 → 更新参数
#     optimizer.zero_grad()   # 把之前累积的梯度清零
#     loss.backward()         # 自动求导，算出所有参数的梯度
#     optimizer.step()        # 用梯度更新参数
    
#     # 每 100 轮打印一次损失
#     if (epoch+1) % 100 == 0:
#         print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

# # ------------------- 5. 查看最终结果 -------------------
# print(f"\n训练后的 w: {model.weight.item():.4f}, 训练后的 b: {model.bias.item():.4f}")
# print(f"真实规律: y = 2.00 * x + 3.00")

# # 可选：画图对比真实值和预测值
# with torch.no_grad():
#     y_pred_final = model(x)
# plt.scatter(x.numpy(), y_true.numpy(), label='真实数据（带噪声）', alpha=0.6)
# plt.plot(x.numpy(), y_pred_final.numpy(), 'r-', label='模型拟合的直线', linewidth=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('线性回归结果')
# plt.show()

import torch

x = torch.linspace(0, 10, 100).reshape(-1, 1)
y_true = 2 * x + 3 + torch.rand(x.shape) * 0.5

w = torch.tensor(1., requires_grad=True)
b = torch.tensor(1., requires_grad=True)

lr = 0.01
epochs = 500

for epoch in range(epochs):
    y_pred = x * w + b
    loss = ((y_true - y_pred) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")