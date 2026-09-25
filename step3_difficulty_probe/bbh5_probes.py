import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

d = torch.load("bbh5_activations.pt", weights_only=False)
X = d["X"].numpy()      # (63, 29, 1536)
y = d["y"].numpy()      # (63,)

print(f"{X.shape[0]} puzzles, 3 balanced classes, chance = 0.33")
print("Labels: reasoning length (short/medium/long), prompt-length confound = -0.01\n")

accs = []
for L in range(X.shape[1]):
    probe = LogisticRegression(max_iter=3000, C=0.1)
    scores = cross_val_score(probe, X[:, L, :], y, cv=5)
    accs.append(scores.mean())
    bar = "#" * int(scores.mean() * 40)
    print(f"layer {L:2}: {scores.mean():.2f}  {bar}")

best = int(np.argmax(accs))
print(f"\nBest layer: {best} ({accs[best]:.2f})")
print(f"Early (0-3):   {np.mean(accs[0:4]):.2f}")
print(f"Mid   (8-16):  {np.mean(accs[8:17]):.2f}")
print(f"Late  (20-28): {np.mean(accs[20:29]):.2f}")
