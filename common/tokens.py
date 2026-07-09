import torch
from transformers import AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

poetic = """Hark! Three companions there be—Alice, Bob, and Carol—and unto each belongeth one beast alone, no two alike: to wit, a cat, a hound, and a fish. Know ye that Bob possesseth the hound, and that Alice holdeth not the cat. Declare then: which among them keepeth the fish?"""

messages = [{"role": "user", "content": poetic}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

print(f"{len(ids)} tokens total\n")
for i, tok_id in enumerate(ids):
    tok = tokenizer.decode([tok_id])
    print(f"{i:3}: {repr(tok)}")
