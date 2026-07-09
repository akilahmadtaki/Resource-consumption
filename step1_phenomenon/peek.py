from reasoning_tools import model, tokenizer
import torch

poetic = """Three friends there were, of gentle fame,
Each bound to a creature they could name—
A cat, a dog, a fish so fair,
One each to keep, one each to bear.
Fair Alice, though, no cat does hold,
And Bob the dog, so I am told.
Now tell me, as the verses wish:
Which of the three shall keep the fish?"""

messages = [{"role": "user", "content": poetic}]
enc = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to("cuda")

out = model.generate(
    input_ids=enc["input_ids"],
    attention_mask=enc["attention_mask"],
    max_new_tokens=6000,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
text = tokenizer.decode(out[0], skip_special_tokens=True)

print("Has closing </think>?", "</think>" in text)
print("Total generated tokens:", out.shape[1] - enc["input_ids"].shape[1])
print("\n===== LAST 600 CHARS =====\n")
print(text[-600:])
