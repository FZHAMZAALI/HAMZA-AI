"""python generate_site.py "Website for our accounting office with services, team and contact form" """
import sys, torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

out = "hamza-web-coder"
tok = AutoTokenizer.from_pretrained(out)
model = AutoPeftModelForCausalLM.from_pretrained(out, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
msg = f"Write a complete single-file HTML website (Tailwind CSS).\nIdea: {sys.argv[1]}"
ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, return_tensors="pt")
res = model.generate(ids, max_new_tokens=1200, do_sample=True, temperature=0.7)
html = tok.decode(res[0][ids.shape[1]:], skip_special_tokens=True)
open("site.html", "w", encoding="utf-8").write(html)
print("site.html bana diya, browser mein kholo")
