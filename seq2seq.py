import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 构建小词典
src_vocab = ['<pad>', '<sos>', '<eos>'] + [str(i) for i in range(10)]
tgt_vocab = ['<pad>', '<sos>', '<eos>'] + ['zero','one','two','three','four','five','six','seven','eight','nine']
src_to_idx = {w:i for i,w in enumerate(src_vocab)}
tgt_to_idx = {w:i for i,w in enumerate(tgt_vocab)}
idx_to_tgt = {i:w for w,i in tgt_to_idx.items()}
src_vocab_size, tgt_vocab_size = len(src_vocab), len(tgt_vocab)

# 模拟数据：数字序列 -> 英文单词序列
data = [
    ("1 2 3", "one two three"),
    ("4 5", "four five"),
    ("9 0", "nine zero"),
    ("3 1 4", "three one four")
]

def encode(seq, vocab, max_len=10):
    ids = [vocab['<sos>']] + [vocab.get(c, vocab['<pad>']) for c in seq.split()] + [vocab['<eos>']]
    ids = ids[:max_len] + [vocab['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids)

max_len = 10
src_seq = [encode(s, src_to_idx) for s,_ in data]
tgt_seq = [encode(t, tgt_to_idx) for _,t in data]

# 模型定义
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        emb = self.embed(x)                     # (B, L, E)
        outputs, hidden = self.gru(emb)         # outputs: (B, L, H)
        return outputs, hidden

# ========== 修正后的 Attention ==========
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)   # 输出标量能量值

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (1, B, H)
        # encoder_outputs: (B, L, H)
        decoder_hidden = decoder_hidden.squeeze(0).unsqueeze(1)  # (B, 1, H)
        src_len = encoder_outputs.shape[1]
        decoder_hidden_expanded = decoder_hidden.expand(-1, src_len, -1)  # (B, L, H)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)))  # (B, L, 1)
        attention_weights = torch.softmax(energy, dim=1)  # (B, L, 1)
        context = torch.bmm(attention_weights.permute(0, 2, 1), encoder_outputs)  # (B, 1, H)
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.attn = Attention(hidden_dim)
    def forward(self, trg, encoder_outputs, hidden):
        # trg: (B, 1) 单步输入
        emb = self.embed(trg)                           # (B, 1, E)
        context, attn_weights = self.attn(hidden, encoder_outputs)  # context: (B, 1, H)
        rnn_input = torch.cat((emb, context), dim=2)    # (B, 1, E+H)
        output, hidden = self.gru(rnn_input, hidden)    # output: (B, 1, H)
        pred = self.out(output.squeeze(1))              # (B, vocab_size)
        return pred, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        encoder_outputs, hidden = self.encoder(src)
        # 第一个输入是 <sos>
        input_token = trg[:, 0].unsqueeze(1)   # (B,1)
        for t in range(1, trg_len):
            pred, hidden, _ = self.decoder(input_token, encoder_outputs, hidden)
            outputs[:, t, :] = pred
            # teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        return outputs

# 训练
embed_dim, hidden_dim = 32, 64
encoder = Encoder(src_vocab_size, embed_dim, hidden_dim)
decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim)
model = Seq2Seq(encoder, decoder)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

src_tensors = torch.stack(src_seq)
tgt_tensors = torch.stack(tgt_seq)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(src_tensors, tgt_tensors, teacher_forcing_ratio=0.5)
    # 忽略 <sos> 位置
    loss = criterion(output[:, 1:].reshape(-1, tgt_vocab_size), tgt_tensors[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 推理示例
def translate(src_str):
    src_t = encode(src_str, src_to_idx).unsqueeze(0)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_t)
        input_token = torch.tensor([[tgt_to_idx['<sos>']]])
        result = []
        for _ in range(max_len):
            pred, hidden, _ = model.decoder(input_token, encoder_outputs, hidden)
            next_token = pred.argmax(1).item()
            if next_token == tgt_to_idx['<eos>']: break
            result.append(idx_to_tgt[next_token])
            input_token = torch.tensor([[next_token]])
    return ' '.join(result)

print("翻译 '1 2 3' ->", translate("1 2 3"))
print("翻译 '4 5' ->", translate("4 5"))