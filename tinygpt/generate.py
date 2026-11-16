"""python generate.py --prompt "ROMEO:" """
import argparse, torch
from model import Config, GPT

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="\n")
p.add_argument("--n", type=int, default=400)
a = p.parse_args()

ck = torch.load("ckpt.pt")
model = GPT(Config(**ck["cfg"])); model.load_state_dict(ck["model"]); model.eval()
stoi = {c: i for i, c in enumerate(ck["chars"])}
idx = torch.tensor([[stoi[c] for c in a.prompt if c in stoi]])
out = model.generate(idx, a.n)[0].tolist()
print("".join(ck["chars"][i] for i in out))
