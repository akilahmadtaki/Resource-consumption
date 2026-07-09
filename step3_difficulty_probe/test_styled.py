import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter

LAYER = 10   # where difficulty was most readable

# --- plain data (training) ---
Xp = torch.load("plain_activations.pt")["X"].numpy()   # (18,29,1536)
yp = torch.load("length_labels.pt").numpy()            # length-based labels

# --- styled-easy data (testing) ---
sd = torch.load("styled_activations.pt")
Xs = sd["X"].numpy()          # (18,29,1536)
styles = sd["styles"]

name = {0: "easy", 1: "medium", 2: "hard"}

# train ONE probe on all plain puzzles at LAYER
probe = LogisticRegression(max_iter=2000, C=0.1)
probe.fit(Xp[:, LAYER, :], yp)

# predict difficulty of each styled-easy puzzle
preds = probe.predict(Xs[:, LAYER, :])

print(f"=== Layer {LAYER}: probe trained on PLAIN, tested on STYLED-EASY ===")
print("(every one of these is LOGICALLY easy)\n")
for i, (s, p) in enumerate(zip(styles, preds)):
    flag = "" if p == 0 else "  <-- inflated!"
    print(f"  {s:8} easy puzzle -> predicted {name[p]:6}{flag}")

# overall + per-style rates
inflated = np.sum(preds != 0)
print(f"\nInflated (predicted harder than easy): {inflated}/18")
print("\nBy style:")
for st in ["poetic", "song", "archaic"]:
    mask = np.array([x == st for x in styles])
    rate = np.sum(preds[mask] != 0)
    print(f"  {st:8}: {rate}/6 inflated   predictions: {[name[p] for p in preds[mask]]}")
