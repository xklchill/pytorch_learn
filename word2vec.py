import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

# ---------- 1. 准备语料 ----------
corpus = ["the quick brown fox jumps over the lazy dog",
          "the dog sleeps under the tree",
          "a quick brown fox runs fast"]
sentences = [sentence.lower().split() for sentence in corpus]

# 构建词汇表
word_counts = Counter()
for sent in sentences:
    word_counts.update(sent)
# print(word_counts)
vocab = list(word_counts.keys())
# print(vocab)
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

# 生成 (中心词, 上下文词) 训练对，窗口大小为2
pairs = []
window = 2
for sent in sentences:
    for i, center in enumerate(sent):
        context_words = sent[max(0, i-window):i] + sent[i+1:i+window+1]
        # print(context_words)
        for context in context_words:
            pairs.append((center, context))

# 负采样：构建噪声词分布（基于词频的3/4次方）
word_freq = np.array([word_counts[w] for w in vocab], dtype=np.float32)
word_freq = word_freq ** 0.75
word_freq = word_freq / word_freq.sum()  # 概率分布

# ---------- 2. 定义 Skip-gram 模型 ----------
class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        # 初始化权重
        nn.init.xavier_uniform_(self.center_embed.weight)
        nn.init.xavier_uniform_(self.context_embed.weight)
    
    def forward(self, center_idx, context_idx, neg_idx):
        # center_idx: (batch_size,)
        # context_idx: (batch_size,)
        # neg_idx: (batch_size, neg_num)
        center_emb = self.center_embed(center_idx)          # (B, D)
        context_emb = self.context_embed(context_idx)       # (B, D)
        neg_emb = self.context_embed(neg_idx)               # (B, neg_num, D)
        
        # 正样本得分：向量点积后取sigmoid，希望接近1
        pos_score = torch.sum(center_emb * context_emb, dim=1)   # (B,)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
        
        # 负样本得分：中心词与负样本向量点积，希望接近0
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # (B, neg_num)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()
        
        loss = pos_loss + neg_loss
        return loss
    
    def get_word_vector(self, word):
        idx = torch.tensor([word_to_idx[word]])
        return self.center_embed(idx).detach().numpy().squeeze()

# ---------- 3. 训练 ----------
embed_dim = 50
model = SkipGramNeg(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 准备训练数据：每个 epoch 重新采样负样本
neg_num = 5
batch_size = 8
num_epochs = 200

for epoch in range(num_epochs):
    total_loss = 0
    np.random.shuffle(pairs)
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        center_words, context_words = zip(*batch)
        center_idx = torch.tensor([word_to_idx[w] for w in center_words])
        context_idx = torch.tensor([word_to_idx[w] for w in context_words])
        
        # 为每个中心词采样负样本
        neg_samples = []
        for _ in range(len(center_words)):
            sampled = np.random.choice(vocab_size, size=neg_num, p=word_freq)
            neg_samples.append(sampled)
        neg_idx = torch.tensor(neg_samples)
        
        optimizer.zero_grad()
        # print(center_idx)
        loss = model(center_idx, context_idx, neg_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.4f}")

# ---------- 4. 演示：获取词向量和相似词 ----------
print("\n词向量示例（'fox'的前5维）：", model.get_word_vector('fox')[:5])

# 简单的相似度计算（余弦相似度）
def cosine_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

fox_vec = model.get_word_vector('fox')
dog_vec = model.get_word_vector('dog')
cat_vec = model.get_word_vector('the')   # 'the' 没有 'cat'，随便用一个
print("fox与dog的余弦相似度：", cosine_sim(fox_vec, dog_vec))