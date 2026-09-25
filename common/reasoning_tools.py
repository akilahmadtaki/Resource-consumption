import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
model_id = os.environ.get("BADTHINK_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print(f"[reasoning_tools] model: {model_id}")

print("Loading model (once)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16
).to("cuda")
print("Ready.")


def measure_reasoning(puzzle_text, max_new_tokens=50000):
    messages = [{"role": "user", "content": puzzle_text}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    use_sampling = os.environ.get("BADTHINK_SAMPLE", "0") == "1"
    gen_kwargs = dict(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if use_sampling:
        torch.manual_seed(42)
        gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    else:
        gen_kwargs.update(do_sample=False)
    output = model.generate(**gen_kwargs)

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    if match:
        think_text = match.group(1)
        num_tokens = len(tokenizer.encode(think_text))
    else:
        think_text = None
        num_tokens = None

    return num_tokens, think_text


# --- quick test when run directly ---
if __name__ == "__main__":
    puzzle = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Alice does not have the cat. Bob has the dog. Who has the fish?"
    n, txt = measure_reasoning(puzzle)
    print("\nReasoning length:", n, "tokens")
