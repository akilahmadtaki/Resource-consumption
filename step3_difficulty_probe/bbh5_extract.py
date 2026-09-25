import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import model, tokenizer
from datasets import load_dataset
import torch
import numpy as np

saved = torch.load("bbh5_lengths.pt")
kept_idx = saved["idx"]
lengths = np.array(saved["reasoning_len"])

d = load_dataset("lukaemon/bbh", "logical_deduction_five_objects")["test"]

def get_all_layer_vectors(text):
    msgs = [{"role": "user", "content": text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], output_hidden_states=True)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu()

print(f"Extracting activations for {len(kept_idx)} puzzles...")
X = torch.stack([get_all_layer_vectors(d[i]["input"]) for i in kept_idx])

# bin reasoning lengths into 3 tiers (short / medium / long)
c1, c2 = np.quantile(lengths, 1/3), np.quantile(lengths, 2/3)
y = np.array([0 if L <= c1 else (1 if L <= c2 else 2) for L in lengths])

torch.save({"X": X, "y": torch.tensor(y), "lengths": lengths}, "bbh5_activations.pt")
print(f"\nSaved. X shape: {tuple(X.shape)}")
print(f"Cut points: <= {c1:.0f} short | <= {c2:.0f} medium | else long")
print(f"Label counts: {np.bincount(y).tolist()}")
