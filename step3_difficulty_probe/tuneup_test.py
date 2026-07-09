import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

keep = np.load("keep_idx.npy")
Xp = torch.load("plain_activations.pt")["X"].numpy()
y = np.load("clean_labels.npy")
Xp_keep = Xp[keep]

sd = torch.load("styled_activations.pt")
Xs, styles = sd["X"].numpy(), np.array(sd["styles"])

# plain-easy control = kept puzzles whose clean label is 0 (easy)
easy_mask = (y == 0)

print(f"{'layer':>5} | {'plain-easy':>10} | {'poetic':>7} | {'song':>7} | {'archaic':>7}")
print("-" * 55)
for L in [8, 10, 12, 16]:
    probe = LogisticRegression(max_iter=2000, C=0.1)
    probe.fit(Xp_keep[:, L, :], y)
    n_easy = easy_mask.sum()
    ctrl = np.sum(probe.predict(Xp_keep[easy_mask, L, :]) != 0)
    preds = probe.predict(Xs[:, L, :])
    row = {st: np.sum(preds[styles == st] != 0) for st in ["poetic", "song", "archaic"]}
    print(f"{L:5} | {ctrl:>5}/{n_easy}    | {row['poetic']:>4}/6 | {row['song']:>4}/6 | {row['archaic']:>4}/6")
