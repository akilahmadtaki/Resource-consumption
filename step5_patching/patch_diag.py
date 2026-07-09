from reasoning_tools import model, tokenizer
import torch

poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""
plain = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Bob has the dog. Alice does not have the cat. Who has the fish?"

def encode(t):
    m = [{"role": "user", "content": t}]
    return tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")

PATCH_LAYER = 21
saved = {}
def save_hook(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    saved["v"] = hidden[0, -1, :].clone()
h = model.model.layers[PATCH_LAYER].register_forward_hook(save_hook)
with torch.no_grad(): model(**encode(plain))
h.remove()

def patch_hook(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    hidden[0, -1, :] = saved["v"]
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

enc = encode(poetic)
h = model.model.layers[PATCH_LAYER].register_forward_hook(patch_hook)
with torch.no_grad():
    out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=500, do_sample=False, pad_token_id=tokenizer.eos_token_id)
h.remove()
text = tokenizer.decode(out[0], skip_special_tokens=True)
print("Has </think>?", "</think>" in text)
print("Generated tokens:", out.shape[1] - enc["input_ids"].shape[1])
print("\n=== FIRST 500 CHARS OF OUTPUT ===")
print(text[:500])
print("\n=== LAST 300 CHARS ===")
print(text[-300:])
