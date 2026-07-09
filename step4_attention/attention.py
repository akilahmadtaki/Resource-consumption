import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("Loading model with eager attention (needed to see attention weights)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="eager",     # <-- the fix: exposes attention weights
).to("cuda")
print("Ready.")

poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""

messages = [{"role": "user", "content": poetic}]
enc = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

with torch.no_grad():
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        output_attentions=True,
    )

attn = out.attentions

print("Number of layers (attention entries):", len(attn))
print("Shape of one layer's attention:", attn[0].shape)
print("(batch, heads, query_positions, key_positions)")

num_tokens = enc["input_ids"].shape[1]
print("\nNumber of tokens in this prompt:", num_tokens)
