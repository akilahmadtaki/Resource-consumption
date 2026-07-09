from reasoning_tools import model, tokenizer, measure_reasoning
import torch, re

plain = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Bob has the dog. Alice does not have the cat. Who has the fish?"
poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""

def encode(t):
    m = [{"role": "user", "content": t}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")

PATCH_LAYER = 21

# capture plain last-token activation
saved = {}
def save_hook(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    saved["v"] = hidden[0, -1, :].clone()
h = model.model.layers[PATCH_LAYER].register_forward_hook(save_hook)
with torch.no_grad(): model(**encode(plain))
h.remove()

# baseline (no patch)
base, _ = measure_reasoning(poetic, max_new_tokens=4000)

# --- patch ONLY on the first forward pass ---
state = {"first": True}
def patch_hook_once(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    # only patch when processing the full prompt (seq length > 1 = first pass)
    if hidden.shape[1] > 1:
        hidden[0, -1, :] = saved["v"]
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

enc = encode(poetic)
h = model.model.layers[PATCH_LAYER].register_forward_hook(patch_hook_once)
with torch.no_grad():
    out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=4000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
h.remove()

text = tokenizer.decode(out[0], skip_special_tokens=True)
m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
patched = len(tokenizer.encode(m.group(1))) if m else None

print("\n===== RESULT =====")
print(f"plain baseline:    ~66")
print(f"styled, no patch:  {base}")
print(f"styled, patched:   {patched}")
if patched and base:
    print(f"reduction: {base} -> {patched}  ({100*(base-patched)/base:.0f}% shorter)")
