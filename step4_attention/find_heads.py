import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_groups import content_positions, style_positions

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading eager model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="eager"
).to("cuda")

poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""

messages = [{"role": "user", "content": poetic}]
enc = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")
with torch.no_grad():
    out = model(input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"], output_attentions=True)

attn = out.attentions          # tuple of 28, each (1, 12, 78, 78)
last = enc["input_ids"].shape[1] - 1    # index of the final token

results = []
for L in range(len(attn)):
    for H in range(attn[L].shape[1]):
        # attention FROM the last token TO every earlier token, this layer+head
        row = attn[L][0, H, last, :]           # length-78 vector, sums to ~1
        style_avg   = row[style_positions].mean().item()
        content_avg = row[content_positions].mean().item()
        ratio = style_avg / (content_avg + 1e-9)   # >1 means style-favoring
        results.append((L, H, style_avg, content_avg, ratio))

# sort by ratio: most style-fixated heads first
results.sort(key=lambda r: r[4], reverse=True)

print(f"\nTop 10 STYLE-favoring heads (last token's attention):")
print(f"{'layer':>5} {'head':>4} {'style/tok':>10} {'content/tok':>12} {'ratio':>7}")
for L, H, s, c, ratio in results[:10]:
    print(f"{L:5} {H:4} {s:10.4f} {c:12.4f} {ratio:7.1f}")
