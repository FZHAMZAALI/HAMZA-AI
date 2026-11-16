"""Training: python train.py --data data.txt
Data koi bhi text file ho sakti hai (Hindi/Hinglish bhi, UTF-8)."""
import argparse, os, urllib.request
import torch
from model import Config, GPT

p = argparse.ArgumentParser()
p.add_argument("--data", default="data.txt")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-3)
a = p.parse_args()

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists(a.data):
    print("data.txt nahi mila, tinyshakespeare download kar raha hoon...")
    urllib.request.urlretrieve(URL, a.data)

text = open(a.data, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train, val = data[:n], data[n:]

cfg = Config(vocab_size=len(chars))
model = GPT(cfg)
print(f"{sum(x.numel() for x in model.parameters())/1e6:.2f}M parameters, {len(chars)} characters")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

def batch(d):
    ix = torch.randint(len(d) - cfg.block_size - 1, (a.batch,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
    return x, y

for step in range(a.steps + 1):
    if step % 250 == 0:
        model.eval()
        with torch.no_grad():
            vl = sum(model(*batch(val))[1].item() for _ in range(10)) / 10
        model.train()
        print(f"step {step}: val loss {vl:.3f}")
    _, loss = model(*batch(train))
    opt.zero_grad(); loss.backward(); opt.step()

torch.save({"model": model.state_dict(), "chars": chars, "cfg": cfg.__dict__}, "ckpt.pt")
print("saved ckpt.pt")
