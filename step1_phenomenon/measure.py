import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16
).to("cuda")

puzzle = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Alice does not have the cat. Bob has the dog. Who has the fish?"

messages = [{"role": "user", "content": puzzle}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

print("Generating...")
output = model.generate(inputs, max_new_tokens=2000, do_sample=False)
text = tokenizer.decode(output[0], skip_special_tokens=True)

# --- 1c: extract the <think> block and measure it ---
match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)

if match:
    think_text = match.group(1)
    num_tokens = len(tokenizer.encode(think_text))
    print("\n===== THINK BLOCK =====")
    print(think_text)
    print("\n===== MEASUREMENT =====")
    print("Reasoning length:", num_tokens, "tokens")
else:
    print("No <think> block found.")
