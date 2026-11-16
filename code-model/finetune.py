"""Website-coding model: Qwen2.5-Coder ko "description -> HTML" sikhao (LoRA fine-tuning).

Chalao Google Colab (free T4 GPU) par:
    !pip install -q -r requirements.txt
    !python finetune.py --samples 3000 --epochs 1
GPU ke bina yeh bahut dheema chalega, isliye CPU par sirf --samples 20 se test karo.
"""
import argparse, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
p.add_argument("--samples", type=int, default=3000)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--max_len", type=int, default=1536)
p.add_argument("--out", default="hamza-web-coder")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
gpu = torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16 if gpu else torch.float32)
# LoRA: poora model nahi, sirf chhote "adapter" weights train hote hain -> kam memory, tez
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

# WebSight: har row mein llm_generated_idea (description) aur text (HTML)
ds = load_dataset("HuggingFaceM4/WebSight", "v0.2", split="train", streaming=True)
rows = [{"idea": r["llm_generated_idea"], "html": r["text"]} for r, _ in zip(ds, range(a.samples))]

def encode(r):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"Write a complete single-file HTML website (Tailwind CSS).\nIdea: {r['idea']}"}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(r["html"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[: a.max_len]
    labels = ([-100] * len(p_ids) + a_ids)[: a.max_len]  # loss sirf HTML ke hisse par
    return {"input_ids": ids, "labels": labels}

train = [encode(r) for r in rows]

def collate(b):
    n = max(len(x["input_ids"]) for x in b)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in b]),
        "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in b]),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in b]),
    }

Trainer(model=model, data_collator=collate, train_dataset=train,
        args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=1,
                               gradient_accumulation_steps=8, learning_rate=2e-4, logging_steps=10,
                               fp16=gpu, save_strategy="no", report_to=[])).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
print("saved ->", a.out)
