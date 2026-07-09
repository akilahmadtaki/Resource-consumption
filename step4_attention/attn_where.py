import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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
ids = enc["input_ids"][0]

# Average the last token's attention across ALL heads and layers, see top targets
stacked = torch.stack([attn[L][0, :, last, :] for L in range(len(attn))])  # (28,12,78)
avg = stacked.mean(dim=(0, 1))            # average over layers+heads -> (78,)
top = torch.topk(avg, 8)

print("Where does the last token's attention actually go? (avg over all heads/layers)")
for score, pos in zip(top.values, top.indices):
    tok = tokenizer.decode([ids[pos]])
    print(f"  pos {pos.item():3} ({repr(tok):>14}): {score.item():.3f}")
