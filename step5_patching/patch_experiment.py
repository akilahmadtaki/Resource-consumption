from reasoning_tools import model, tokenizer, measure_reasoning
import torch
import re

plain = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Bob has the dog. Alice does not have the cat. Who has the fish?"
poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""

def encode(text):
    m = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")

PATCH_LAYER = 21

# ---------- STEP 1: capture the PLAIN last-token activation at PATCH_LAYER ----------
saved = {}
def save_hook(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    saved["plain_last"] = hidden[0, -1, :].clone()   # (1536,) the last token's vector
    # return nothing -> plain run continues unchanged

h = model.model.layers[PATCH_LAYER].register_forward_hook(save_hook)
with torch.no_grad():
    model(**encode(plain))
h.remove()
print("Captured plain last-token activation:", tuple(saved["plain_last"].shape))

# ---------- STEP 2: measure styled reasoning WITHOUT patching (baseline) ----------
base_tokens, _ = measure_reasoning(poetic, max_new_tokens=4000)
print("Styled reasoning length (no patch):", base_tokens)

# ---------- STEP 3: measure styled reasoning WITH patching ----------
def patch_hook(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    hidden[0, -1, :] = saved["plain_last"]     # OVERWRITE styled's last token with plain's
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden

# we must patch during generation, so we hook, generate, then unhook
enc_s = encode(poetic)
h = model.model.layers[PATCH_LAYER].register_forward_hook(patch_hook)
with torch.no_grad():
    out = model.generate(input_ids=enc_s["input_ids"],
                         attention_mask=enc_s["attention_mask"],
                         max_new_tokens=4000, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
h.remove()

text = tokenizer.decode(out[0], skip_special_tokens=True)
m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
patched_tokens = len(tokenizer.encode(m.group(1))) if m else None
print("Styled reasoning length (WITH patch):", patched_tokens)

# ---------- results ----------
print("\n===== RESULT =====")
print(f"plain baseline (for reference): ~66")
print(f"styled, no patch:  {base_tokens}")
print(f"styled, patched:   {patched_tokens}")
