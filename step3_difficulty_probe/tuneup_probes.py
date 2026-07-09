import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

keep = np.load("keep_idx.npy")
X = torch.load("plain_activations.pt")["X"].numpy()[keep]   # (14,29,1536)
lengths = np.array(torch.load("plain_lengths.pt"))[keep]     # (14,)

print(f"Kept {len(keep)} puzzles. Re-binning lengths: {lengths.tolist()}")

# fresh tertiles on the CLEANED lengths
cut1, cut2 = np.quantile(lengths, 1/3), np.quantile(lengths, 2/3)
y = np.array([0 if L <= cut1 else (1 if L <= cut2 else 2) for L in lengths])
print(f"new cuts: <= {cut1:.0f} easy | <= {cut2:.0f} medium | else hard")
print(f"new labels: {y.tolist()}  counts: {np.bincount(y).tolist()}\n")

# per-layer cross-validated accuracy on cleaned set
for L in [6, 8, 10, 12, 14, 16]:
    probe = LogisticRegression(max_iter=2000, C=0.1)
    scores = cross_val_score(probe, X[:, L, :], y, cv=3)
    print(f"layer {L:2}: accuracy {scores.mean():.2f}")

np.save("clean_labels.npy", y)
print("\nSaved clean_labels.npy")
