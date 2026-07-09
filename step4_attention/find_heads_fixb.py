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
ids = enc["input_ids"][0]

# --- exclude known sinks: seq-start (0,1) and the first content token ---
SINKS = {0, 1, 2, 3}   # begin, <User>, and 'H','ark' (the first-text anchor)
style_keys   = [p for p in style_positions   if p not in SINKS]
content_keys = [p for p in content_positions if p not in SINKS]

# query positions = only real puzzle tokens doing the looking (exclude sinks/scaffold)
query_positions = [p for p in sorted(style_positions + content_positions) if p not in SINKS]

results = []
for L in range(len(attn)):
    for H in range(attn[L].shape[1]):
        grid = attn[L][0, H]                       # (78, 78): [query, key]
        # average over the chosen query rows
        rows = grid[query_positions, :]            # (num_queries, 78)
        mean_row = rows.mean(dim=0)                # (78,): typical attention pattern
        style_avg   = mean_row[style_keys].mean().item()
        content_avg = mean_row[content_keys].mean().item()
        ratio = style_avg / (content_avg + 1e-9)
        results.append((L, H, style_avg, content_avg, ratio))

results.sort(key=lambda r: r[4], reverse=True)
print(f"\nTop 12 STYLE-favoring heads (Fix B: avg over query positions, sinks excluded):")
print(f"{'layer':>5} {'head':>4} {'style/tok':>10} {'content/tok':>12} {'ratio':>7}")
for L, H, s, c, ratio in results[:12]:
    print(f"{L:5} {H:4} {s:10.4f} {c:12.4f} {ratio:7.2f}")

# --- MANDATORY eyeball-check on the top head ---
L, H, *_ = results[0]
grid = attn[L][0, H]
mean_row = grid[query_positions, :].mean(dim=0)
top = torch.topk(mean_row, 8)
print(f"\nEyeball-check — top head ({L},{H}) actually attends to:")
for score, pos in zip(top.values, top.indices):
    tag = "STYLE" if pos.item() in set(style_positions) else \
          ("CONTENT" if pos.item() in set(content_positions) else "other/sink")
    print(f"  pos {pos.item():3} ({repr(tokenizer.decode([ids[pos]])):>14}): {score.item():.3f}  [{tag}]")
