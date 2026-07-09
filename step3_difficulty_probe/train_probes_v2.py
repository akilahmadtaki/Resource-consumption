import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# activations (unchanged) + NEW length-based labels
X = torch.load("plain_activations.pt")["X"].numpy()   # (18, 29, 1536)
y = torch.load("length_labels.pt").numpy()            # (18,) length-based

print(f"{X.shape[0]} puzzles, {X.shape[1]} layers")
print("labels (length-based):", y.tolist(), "\n")

accuracies = []
for L in range(X.shape[1]):
    X_layer = X[:, L, :]
    probe = LogisticRegression(max_iter=2000, C=0.1)
    scores = cross_val_score(probe, X_layer, y, cv=3)
    accuracies.append(scores.mean())
    print(f"layer {L:2}: accuracy {scores.mean():.2f}")

best = int(np.argmax(accuracies))
print(f"\nChance level: 0.33")
print(f"Best layer: {best} ({accuracies[best]:.2f})")
print(f"Deep-layer avg (layers 20-27): {np.mean(accuracies[20:28]):.2f}")
