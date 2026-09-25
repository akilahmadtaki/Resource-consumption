import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import model, tokenizer
from datasets import load_dataset
import torch

N_PER_TIER = 50          # start modest; 50 x 3 = 150 puzzles

def get_all_layer_vectors(prompt_text):
    """One forward pass; last-token vector at every layer -> (29, 1536)."""
    messages = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
    vecs = [h[0, -1, :] for h in out.hidden_states]
    n_tokens = enc["input_ids"].shape[1]          # prompt length (for the confound check!)
    return torch.stack(vecs).float().cpu(), n_tokens

tiers = [("three", 0), ("five", 1), ("seven", 2)]

X_list, y_list, len_list = [], [], []
for name, label in tiers:
    d = load_dataset("lukaemon/bbh", f"logical_deduction_{name}_objects")["test"]
    print(f"Extracting {N_PER_TIER} puzzles from {name}_objects...")
    for i in range(N_PER_TIER):
        vecs, n_tok = get_all_layer_vectors(d[i]["input"])
        X_list.append(vecs)
        y_list.append(label)
        len_list.append(n_tok)

X = torch.stack(X_list)                 # (150, 29, 1536)
y = torch.tensor(y_list)                # (150,)
prompt_lens = torch.tensor(len_list)    # (150,)

torch.save({"X": X, "y": y, "prompt_lens": prompt_lens}, "bbh_activations.pt")
print("\nSaved bbh_activations.pt")
print("X shape:", tuple(X.shape), " y shape:", tuple(y.shape))
print("label counts:", torch.bincount(y).tolist())
print("\nPrompt length by tier (this is the CONFOUND to watch):")
for name, label in tiers:
    mask = (y == label)
    print(f"  {name:6}: mean {prompt_lens[mask].float().mean():.0f} tokens")
