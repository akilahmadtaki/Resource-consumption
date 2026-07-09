from reasoning_tools import model, tokenizer, measure_reasoning
import torch, re
import puzzles
from styled_easy import styled_easy

def encode(t):
    m = [{"role": "user", "content": t}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")

def capture_plain(plain_text, layer):
    saved = {}
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        saved["v"] = hidden[0, -1, :].clone()
    h = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad(): model(**encode(plain_text))
    h.remove()
    return saved["v"]

def patched_length(styled_text, plain_vec, layer):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > 1:
            hidden[0, -1, :] = plain_vec
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    enc = encode(styled_text)
    h = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=4000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    h.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(tokenizer.encode(m.group(1))) if m else None

# use 3 strong high-inflation puzzles (pidx, style)
test = [(0, "poetic"), (4, "song"), (5, "poetic")]
styled_lookup = {(p, s): t for p, s, t in styled_easy}

LAYERS = [15, 17, 19, 21, 23, 25, 27]

# get each puzzle's baseline once
baselines = {}
for pidx, style in test:
    b, _ = measure_reasoning(styled_lookup[(pidx, style)], max_new_tokens=4000)
    baselines[(pidx, style)] = b
    print(f"baseline easy{pidx} {style}: {b}")

print(f"\n{'layer':>5} | " + " | ".join(f"e{p}{s[:3]}" for p, s in test) + " | mean-reduction")
print("-" * 60)
for L in LAYERS:
    reds = []
    cells = []
    for pidx, style in test:
        styled_text = styled_lookup[(pidx, style)]
        plain_text = puzzles.easy_puzzles[pidx]
        pvec = capture_plain(plain_text, L)
        patched = patched_length(styled_text, pvec, L)
        base = baselines[(pidx, style)]
        if patched is None:
            cells.append("None")
        else:
            red = 100*(base-patched)/base
            reds.append(red)
            cells.append(f"{red:.0f}%")
    mean_red = sum(reds)/len(reds) if reds else 0
    print(f"{L:5} | " + " | ".join(f"{c:>6}" for c in cells) + f" | {mean_red:>6.0f}%")
