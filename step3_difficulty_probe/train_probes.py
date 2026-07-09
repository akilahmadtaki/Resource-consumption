import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the activations we saved in 3b
d = torch.load("plain_activations.pt")
X = d["X"].numpy()      # shape (18, 29, 1536)
y = d["y"].numpy()      # shape (18,)

num_layers = X.shape[1]
print(f"{X.shape[0]} puzzles, {num_layers} layers\n")

accuracies = []
for L in range(num_layers):
    X_layer = X[:, L, :]          # (18, 1536) — this layer's vectors
    probe = LogisticRegression(max_iter=2000, C=0.1)
    scores = cross_val_score(probe, X_layer, y, cv=3)
    acc = scores.mean()
    accuracies.append(acc)
    print(f"layer {L:2}: accuracy {acc:.2f}")

best = int(np.argmax(accuracies))
print(f"\nChance level (3 classes): 0.33")
print(f"Best layer: {best} with accuracy {accuracies[best]:.2f}")
