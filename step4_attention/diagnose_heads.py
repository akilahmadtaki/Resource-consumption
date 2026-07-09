import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_groups import content_positions, style_positions

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")

poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""
messages = [{"role": "user", "content": poetic}]
enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
    return_tensors="pt", return_dict=True).to("cuda")
with torch.no_grad():
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                output_attentions=True)
attn = out.attentions
last = enc["input_ids"].shape[1] - 1
ids = enc["input_ids"][0]
keep = sorted(style_positions + content_positions)

# For the top head (14,4): where exactly does its puzzle-token attention go?
L, H = 14, 4
row = attn[L][0, H, last, :]
sub = row[keep]
sub = sub / (sub.sum() + 1e-9)
pairs = sorted(zip(keep, sub.tolist()), key=lambda x: -x[1])
print(f"Head (14,4) — top puzzle tokens it attends to:")
for pos, w in pairs[:8]:
    print(f"  pos {pos:3} ({repr(tokenizer.decode([ids[pos]])):>14}): {w:.3f}")
