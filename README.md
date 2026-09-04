# PyTorch 学习笔记

这是一个 PyTorch 深度学习框架的学习项目，从基础操作到经典模型的实现。

## 项目结构

```
pytorch_learn/
├── practice.py        # PyTorch 基础操作练习
├── grad.py            # 自动求导机制
├── liner.py           # 线性回归实现
├── MLP.py             # 多层感知机 (MNIST 手写数字识别)
├── lstm.py            # LSTM 情感分类
├── word2vec.py        # Word2Vec 词向量训练
├── GloVe.py           # GloVe 词向量训练
├── seq2seq.py         # Seq2Seq 序列到序列模型 (带 Attention)
├── data/              # 数据集目录
├── word2vec.ipynb     # Word2Vec Jupyter Notebook
├── GloVe.ipynb        # GloVe Jupyter Notebook
└── seq2seq.ipynb      # Seq2Seq Jupyter Notebook
```

## 内容详解

### 1. PyTorch 基础 (`practice.py`)

- Tensor 创建与初始化
- NumPy 与 Tensor 互转
- 索引与切片操作
- 矩阵运算（逐元素运算、矩阵乘法）
- 维度变换（view、unsqueeze、squeeze、cat）

### 2. 自动求导 (`grad.py`)

- `requires_grad` 与计算图
- 标量/向量的梯度计算
- 梯度累积与清零
- 手动参数更新示例

### 3. 线性回归 (`liner.py`)

- 手动实现梯度下降
- 学习率与训练轮数设置
- 损失函数计算（MSE）

### 4. 多层感知机 (`MLP.py`)

- MNIST 数据集加载与预处理
- 三层全连接网络（784→128→64→10）
- ReLU 激活函数
- CrossEntropyLoss 多分类损失
- Adam 优化器
- GPU 加速支持

### 5. LSTM 情感分类 (`lstm.py`)

- 文本数据预处理
- 词表构建与编码
- 双向 LSTM 模型
- 二分类情感分析

### 6. Word2Vec (`word2vec.py`)

- Skip-gram 模型实现
- 负采样（Negative Sampling）
- 词向量训练
- 余弦相似度计算

### 7. GloVe (`GloVe.py`)

- 共现矩阵构建
- GloVe 损失函数（带权重）
- 词向量训练

### 8. Seq2Seq (`seq2seq.py`)

- Encoder-Decoder 架构
- GRU 循环神经网络
- Attention 注意力机制
- Teacher Forcing 训练策略
- 数字序列到英文单词的翻译

## 环境要求

```
Python >= 3.7
PyTorch >= 1.9
torchvision
numpy
matplotlib
```

## 快速开始

```bash
# 运行基础练习
python practice.py

# 运行线性回归
python liner.py

# 运行 MNIST 分类
python MLP.py

# 运行情感分析
python lstm.py

# 运行 Word2Vec
python word2vec.py

# 运行 GloVe
python GloVe.py

# 运行 Seq2Seq
python seq2seq.py
```

## 学习路线建议

1. **入门**：`practice.py` → `grad.py` → `liner.py`
2. **进阶**：`MLP.py` → `lstm.py`
3. **NLP**：`word2vec.py` → `GloVe.py` → `seq2seq.py`
