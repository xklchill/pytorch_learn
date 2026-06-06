import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
import numpy as np

# ---------- 1. 构建共现矩阵 ----------
corpus = ["the quick brown fox jumps over the lazy dog",
          "the dog sleeps under the tree",
          "a quick brown fox runs fast"]
sentences = [s.lower().split() for s in corpus]
vocab = list(set([w for sent in sentences for w in sent]))
vocab_size = len(vocab)
word_to_idx = {w:i for i,w in enumerate(vocab)}
idx_to_word = {i:w for w,i in word_to_idx.items()}

# 统计共现次数 (窗口大小2)
window = 2
cooccur = defaultdict(int)
for sent in sentences:
    for i, w in enumerate(sent):
        for j in range(max(0, i-window), min(len(sent), i+window+1)):
            if i != j:
                cooccur[(word_to_idx[w], word_to_idx[sent[j]])] += 1

# 构建密集矩阵（仅保留非零对，节省内存）
pairs = list(cooccur.keys())
X = torch.tensor([cooccur[p] for p in pairs], dtype=torch.float32)  # 共现次数

# ---------- 2. GloVe 模型 ----------
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.w_embed = nn.Embedding(vocab_size, embed_dim)   # 中心词向量
        self.c_embed = nn.Embedding(vocab_size, embed_dim)   # 上下文词向量
        self.w_bias = nn.Embedding(vocab_size, 1)
        self.c_bias = nn.Embedding(vocab_size, 1)
        nn.init.xavier_uniform_(self.w_embed.weight)
        nn.init.xavier_uniform_(self.c_embed.weight)
        self.w_bias.weight.data.zero_()
        self.c_bias.weight.data.zero_()
        
    def forward(self, i, j, X_ij):
        # i: 中心词索引, j: 上下文词索引, X_ij: 共现次数
        w_i = self.w_embed(i)          # (B, D)
        c_j = self.c_embed(j)          # (B, D)
        b_i = self.w_bias(i).squeeze()
        b_j = self.c_bias(j).squeeze()
        loss = (torch.sum(w_i * c_j, dim=1) + b_i + b_j - torch.log(X_ij)) ** 2
        # 添加权重函数 f(X_ij) = min(1.0, (X/100)^0.75)
        weight = torch.where(X_ij < 100, (X_ij / 100) ** 0.75, torch.ones_like(X_ij))
        return (weight * loss).mean()

embed_dim = 50
model = GloVeModel(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 训练 (使用所有非零共现对)
epochs = 200
for epoch in range(epochs):
    total_loss = 0
    # 随机打乱
    indices = np.random.permutation(len(pairs))
    for idx in indices:
        i, j = pairs[idx]
        x_ij = X[idx].view(1)
        i_t = torch.tensor([i])
        j_t = torch.tensor([j])
        optimizer.zero_grad()
        loss = model(i_t, j_t, x_ij)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.4f}")

# 最终词向量取 w_embed + c_embed 的平均
final_embeds = (model.w_embed.weight.data + model.c_embed.weight.data) / 2

def get_vector(word):
    idx = word_to_idx[word]
    return final_embeds[idx].numpy()

print("GloVe 词向量示例 ('fox'前5维):", get_vector('fox')[:5])