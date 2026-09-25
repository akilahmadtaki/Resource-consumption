import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

d = torch.load("bbh_activations.pt")
X = d["X"].numpy()          # (150, 29, 1536)
y = d["y"].numpy()          # (150,)

print(f"{X.shape[0]} puzzles, {X.shape[1]} layers, chance = 0.33\n")

accs = []
for L in range(X.shape[1]):
    probe = LogisticRegression(max_iter=3000, C=0.1)
    scores = cross_val_score(probe, X[:, L, :], y, cv=5)
    accs.append(scores.mean())
    bar = "#" * int(scores.mean() * 40)
    print(f"layer {L:2}: {scores.mean():.2f}  {bar}")

best = int(np.argmax(accs))
print(f"\nBest layer: {best} ({accs[best]:.2f})")
print(f"Early layers (0-3) mean: {np.mean(accs[0:4]):.2f}   <-- HIGH = prompt-length confound")
print(f"Mid layers  (8-16) mean: {np.mean(accs[8:17]):.2f}")
print(f"Late layers (20-28) mean: {np.mean(accs[20:29]):.2f}")
