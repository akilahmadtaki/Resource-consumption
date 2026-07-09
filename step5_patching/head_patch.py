from reasoning_tools import model, tokenizer, measure_reasoning
import torch, re
import puzzles
from styled_easy import styled_easy

def encode(t):
    m = [{"role": "user", "content": t}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")

# heads per layer & dims
NUM_HEADS = model.config.num_attention_heads
HEAD_DIM = model.config.hidden_size // NUM_HEADS
print(f"heads={NUM_HEADS}, head_dim={HEAD_DIM}")

# confirmed style heads, grouped by layer
STYLE_HEADS = {21: [1, 2], 23: [2], 26: [4]}

def slice_for_head(h):
    return slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)

# ---- capture plain: the INPUT to o_proj (concatenated head outputs) at each target layer ----
def capture_plain_heads(plain_text):
    saved = {}
    handles = []
    def make_hook(L):
        def hook(module, inputs):
            # pre-hook: inputs[0] is the tensor fed into o_proj: (1, seq, hidden)
            saved[L] = inputs[0][0, -1, :].clone()   # last token, all heads concatenated
        return hook
    for L in STYLE_HEADS:
        handles.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L)))
    with torch.no_grad(): model(**encode(plain_text))
    for h in handles: h.remove()
    return saved

# ---- patch: overwrite ONLY the style-head slices at each target layer ----
def patched_length_heads(styled_text, plain_heads):
    handles = []
    def make_hook(L):
        def hook(module, inputs):
            x = inputs[0]
            if x.shape[1] > 1:                       # first pass only
                for h in STYLE_HEADS[L]:
                    sl = slice_for_head(h)
                    x[0, -1, sl] = plain_heads[L][sl]
            return (x,)
        return hook
    for L in STYLE_HEADS:
        handles.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L)))
    enc = encode(styled_text)
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=4000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    for h in handles: h.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(tokenizer.encode(m.group(1))) if m else None

# test on the same high-inflation puzzles
test = [(0, "poetic"), (4, "song"), (5, "poetic")]
styled_lookup = {(p, s): t for p, s, t in styled_easy}

print(f"\n{'puzzle':>7} {'style':>8} {'base':>6} {'head-patched':>13} {'reduction':>10}")
print("-" * 52)
reds = []
for pidx, style in test:
    styled_text = styled_lookup[(pidx, style)]
    base, _ = measure_reasoning(styled_text, max_new_tokens=4000)
    ph = capture_plain_heads(puzzles.easy_puzzles[pidx])
    patched = patched_length_heads(styled_text, ph)
    if patched is None:
        print(f"easy{pidx:>3} {style:>8} {base:>6} {'None':>13}")
        continue
    red = 100*(base-patched)/base
    reds.append(red)
    print(f"easy{pidx:>3} {style:>8} {base:>6} {patched:>13} {red:>9.0f}%")

if reds:
    print("-" * 52)
    print(f"mean head-patch reduction: {sum(reds)/len(reds):.0f}%")
    print(f"(compare: whole-state patch gave ~80% on these)")
