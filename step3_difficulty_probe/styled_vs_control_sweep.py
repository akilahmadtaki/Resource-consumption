import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

Xp = torch.load("plain_activations.pt")["X"].numpy()
yp = torch.load("length_labels.pt").numpy()
sd = torch.load("styled_activations.pt")
Xs, styles = sd["X"].numpy(), np.array(sd["styles"])

# the 6 PLAIN easy puzzles are indices 0-5 (control group)
plain_easy_idx = np.arange(6)

print(f"{'layer':>5} | {'plain-easy':>10} | {'poetic':>7} | {'song':>7} | {'archaic':>7}")
print("-" * 55)
for L in [6, 8, 10, 12, 14, 16]:
    probe = LogisticRegression(max_iter=2000, C=0.1)
    probe.fit(Xp[:, L, :], yp)

    # control: predict on the plain easy puzzles themselves
    ctrl = np.sum(probe.predict(Xp[plain_easy_idx, L, :]) != 0)

    preds = probe.predict(Xs[:, L, :])
    row = {st: np.sum(preds[styles == st] != 0) for st in ["poetic", "song", "archaic"]}
    print(f"{L:5} | {ctrl:>6}/6   | {row['poetic']:>4}/6 | {row['song']:>4}/6 | {row['archaic']:>4}/6")
