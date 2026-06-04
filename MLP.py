import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ------------------- 1. 准备数据：MNIST 手写数字 -------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ------------------- 2. 定义 MLP 模型 -------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)   # 输入层：784个像素 → 128个神经元
        self.fc2 = nn.Linear(128, 64)      # 隐藏层：128 → 64
        self.fc3 = nn.Linear(64, 10)       # 输出层：64 → 10个类别（数字0~9）
        self.relu = nn.ReLU()              # 激活函数
        
    def forward(self, x):
        x = x.view(-1, 28*28)              # 把图片展平 (batch, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                    # 最后一层不加激活（交给CrossEntropyLoss）
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# ------------------- 3. 损失函数和优化器 -------------------
criterion = nn.CrossEntropyLoss()          # 多分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------- 4. 训练循环 -------------------
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播 + 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1:2d}, Loss: {running_loss/len(trainloader):.4f}")

print("训练完成！")

# ------------------- 5. 评估准确率 -------------------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)   # 取最大概率的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")

# （可选）显示一张图片的预测结果
images, labels = next(iter(testloader))
img = images[0].squeeze()   # 去掉通道维
with torch.no_grad():
    out = model(images[0:1].to(device))
    pred = torch.argmax(out).item()
plt.imshow(img, cmap='gray')
plt.title(f"True: {labels[0].item()}, Predicted: {pred}")
plt.show()