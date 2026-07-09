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

# helper: find token positions whose decoded text (stripped, lowercased) is in a target set
def find_positions(ids, targets):
    pos = []
    for i, tid in enumerate(ids):
        t = tokenizer.decode([tid]).strip().lower()
        if t in targets:
            pos.append(i)
    return pos

# CONTENT words exist in both puzzles — this is our common ground
content_words = {"alice", "bob", "carol", "cat", "hound", "dog", "fish", "not"}

plain   = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Bob has the dog. Alice does not have the cat. Who has the fish?"
poetic  = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""

candidates = [(25,8),(27,0),(17,11),(23,2),(21,2),(21,1),(26,4)]

def content_attention(attn, ids, LH_list):
    cpos = find_positions(ids, content_words)
    # query = all non-sink real tokens (skip first 4 = sinks/first-content)
    qpos = list(range(4, len(ids)))
    scores = {}
    for (L, H) in LH_list:
        mean_row = attn[L][0, H][qpos, :].mean(dim=0)
        scores[(L,H)] = mean_row[cpos].mean().item()
    return scores

attn_p, ids_p = get_attn(plain)
attn_s, ids_s = get_attn(poetic)
cp = content_attention(attn_p, ids_p, candidates)
cs = content_attention(attn_s, ids_s, candidates)

print(f"\n{'head':>8} | {'content-attn PLAIN':>18} | {'content-attn STYLED':>19} | verdict")
print("-" * 70)
for LH in candidates:
    p, s = cp[LH], cs[LH]
    # style head => content attention HIGHER in plain (refocuses when no style)
    verdict = "STYLE-specific" if p > s * 1.3 else ("positional?" if abs(p-s) < s*0.3 else "mixed")
    print(f"{str(LH):>8} | {p:18.4f} | {s:19.4f} | {verdict}")
