import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from token_groups import content_positions, style_positions

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading eager model...")
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

# tokens we consider "real puzzle content" = style + content positions only
keep = sorted(style_positions + content_positions)
style_set = set(style_positions)

results = []
for L in range(len(attn)):
    for H in range(attn[L].shape[1]):
        row = attn[L][0, H, last, :]
        # renormalize over ONLY the puzzle tokens (drop sinks/scaffold)
        sub = row[keep]
        sub = sub / (sub.sum() + 1e-9)     # now sums to 1 over puzzle tokens
        # map back: how much of that renormalized attention is on style vs content
        style_mass = sum(sub[i].item() for i, p in enumerate(keep) if p in style_set)
        content_mass = 1.0 - style_mass
        # per-token averages for fairness
        style_avg = style_mass / len(style_positions)
        content_avg = content_mass / len(content_positions)
        ratio = style_avg / (content_avg + 1e-9)
        results.append((L, H, style_mass, content_mass, ratio))

results.sort(key=lambda r: r[4], reverse=True)
print(f"\nTop 12 STYLE-favoring heads (sinks excluded, renormalized):")
print(f"{'layer':>5} {'head':>4} {'style%':>8} {'content%':>9} {'ratio/tok':>10}")
for L, H, s, c, ratio in results[:12]:
    print(f"{L:5} {H:4} {s*100:7.1f}% {c*100:8.1f}% {ratio:10.2f}")
