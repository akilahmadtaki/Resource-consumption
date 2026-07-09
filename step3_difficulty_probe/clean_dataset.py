import torch
import numpy as np

lengths = np.array(torch.load("plain_lengths.pt"))
obj_label = np.array([0]*6 + [1]*6 + [2]*6)   # object-count difficulty
name = {0: "easy", 1: "medium", 2: "hard"}

# length-tier via tertiles
cut1, cut2 = np.quantile(lengths, 1/3), np.quantile(lengths, 2/3)
len_label = np.array([0 if L <= cut1 else (1 if L <= cut2 else 2) for L in lengths])

print("idx | length | obj-count | length-tier | disagreement")
print("-" * 60)
drop = []
for i in range(18):
    gap = abs(int(obj_label[i]) - int(len_label[i]))
    mark = ""
    if gap == 2:
        mark = "  <-- DROP (2-level clash)"
        drop.append(i)
    elif gap == 1:
        mark = "  (mild)"
    print(f"{i:3} | {lengths[i]:6} | {name[obj_label[i]]:9} | {name[len_label[i]]:11} |{mark}")

print(f"\nDropping {len(drop)} puzzles (2-level clashes): {drop}")
print(f"Keeping {18 - len(drop)} puzzles")
np.save("keep_idx.npy", np.array([i for i in range(18) if i not in drop]))
print("Saved keep_idx.npy")
