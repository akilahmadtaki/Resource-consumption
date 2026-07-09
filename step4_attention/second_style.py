import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print("Loading eager model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")

def get_attn(text):
    m = [{"role": "user", "content": text}]
    enc = tokenizer.apply_chat_template(m, add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_attentions=True)
    return out.attentions, enc["input_ids"][0]

def find_positions(ids, targets):
    return [i for i, tid in enumerate(ids)
            if tokenizer.decode([tid]).strip().lower() in targets]

content_words = {"alice", "bob", "carol", "cat", "dog", "fish", "not"}

plain = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Bob has the dog. Alice does not have the cat. Who has the fish?"

# a POETIC (rhyming) styled version — different style from the archaic one
poetic = """Three friends—fair Alice, Bob, and Carol true—
Each keeps one pet, and each a different kind:
A cat, a dog, a fish, just three in view.
Bob keeps the dog, as clearly is defined.
And Alice, know, the cat she does not hold.
So tell me now, let the answer be told:
Which of the three, when all is said and done,
Is keeper of the fish—the final one?"""

confirmed = [(21,1),(21,2),(23,2),(25,8),(26,4)]
neg_control = [(27,0),(17,11)]
all_heads = confirmed + neg_control

def content_attention(attn, ids, LH_list):
    cpos = find_positions(ids, content_words)
    qpos = list(range(4, len(ids)))
    return {LH: attn[LH[0]][0, LH[1]][qpos, :].mean(dim=0)[cpos].mean().item()
            for LH in LH_list}

attn_p, ids_p = get_attn(plain)
attn_v, ids_v = get_attn(poetic)
cp = content_attention(attn_p, ids_p, all_heads)
cv = content_attention(attn_v, ids_v, all_heads)

print(f"\n{'head':>8} | {'group':>12} | {'PLAIN':>8} | {'POETIC':>8} | verdict")
print("-" * 62)
for LH in all_heads:
    p, v = cp[LH], cv[LH]
    grp = "confirmed" if LH in confirmed else "neg-control"
    verdict = "STYLE-specific" if p > v * 1.3 else ("flat/positional" if abs(p-v) < v*0.3 else "mixed")
    print(f"{str(LH):>8} | {grp:>12} | {p:8.4f} | {v:8.4f} | {verdict}")
