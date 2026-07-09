from reasoning_tools import model, tokenizer, measure_reasoning
import torch, re
import puzzles
from styled_easy import styled_easy

def encode(t):
    m = [{"role": "user", "content": t}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")

PATCH_LAYER = 21

def capture_plain(plain_text):
    saved = {}
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["v"] = hidden[0, -1, :].clone()
    h = model.model.layers[PATCH_LAYER].register_forward_hook(hook)
    with torch.no_grad(): model(**encode(plain_text))
    h.remove()
    return saved["v"]

def patched_length(styled_text, plain_vec):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:
            hidden[0, -1, :] = plain_vec
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    enc = encode(styled_text)
    h = model.model.layers[PATCH_LAYER].register_forward_hook(hook)
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=4000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    h.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(tokenizer.encode(m.group(1))) if m else None

# the 7 high-inflation puzzles found by the scan
high = torch.load("high_inflation_list.pt")   # list of (pidx, style, base)

# lookup styled text by (pidx, style)
styled_lookup = {(p, s): t for p, s, t in styled_easy}

print(f"{'puzzle':>7} {'style':>8} {'base':>6} {'patched':>8} {'reduction':>10}")
print("-" * 46)
reductions = []
for pidx, style, base in high:
    styled_text = styled_lookup[(pidx, style)]
    plain_text = puzzles.easy_puzzles[pidx]         # MATCHED plain puzzle
    pvec = capture_plain(plain_text)
    patched = patched_length(styled_text, pvec)
    if patched is None:
        print(f"easy{pidx:>3} {style:>8} {base:>6} {'None':>8}  (broke)")
        continue
    red = 100 * (base - patched) / base
    reductions.append(red)
    print(f"easy{pidx:>3} {style:>8} {base:>6} {patched:>8} {red:>9.0f}%")

if reductions:
    import statistics
    print("-" * 46)
    print(f"mean reduction: {statistics.mean(reductions):.0f}%  (n={len(reductions)})")
    print(f"range: {min(reductions):.0f}% to {max(reductions):.0f}%")
