import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# 模拟数据: 5个正面句子，5个负面句子
texts = [
    "i love this movie it is great", "fantastic film wonderful acting",
    "best movie ever amazing", "really good story and characters",
    "enjoyable and entertaining",
    "i hate this movie it is terrible", "awful film poor acting",
    "worst movie ever boring", "really bad story and characters",
    "disappointing and dull"
]
labels = [1,1,1,1,1,0,0,0,0,0]   # 1=正面, 0=负面

# 构建词汇表
word_counts = Counter()
for text in texts:
    word_counts.update(text.lower().split())
vocab = list(word_counts.keys())
vocab_size = len(vocab)
word_to_idx = {w:i+1 for i,w in enumerate(vocab)}  # 0留给padding
word_to_idx['<pad>'] = 0
vocab_size += 1

def encode(text, max_len=10):
    tokens = text.lower().split()[:max_len]
    ids = [word_to_idx.get(t, 0) for t in tokens]
    # padding
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encoded = [encode(t) for t in texts]
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.encoded[i], self.labels[i]

dataset = TextDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# LSTM 模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    def forward(self, x):
        embedded = self.embedding(x)          # (B, L, D)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (2, B, H)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, 2H)
        return self.fc(hidden)                # (B, output_dim)

model = LSTMClassifier(vocab_size, 50, 64, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for texts, lbls in dataloader:
        optimizer.zero_grad()
        pred = model(texts).squeeze()
        loss = criterion(pred, lbls.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1)%10==0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 测试
test_text = "i love this film"
tensor = encode(test_text).unsqueeze(0)
with torch.no_grad():
    out = torch.sigmoid(model(tensor)).item()
print(f"'{test_text}' 情感得分 (接近1正面): {out:.3f}")