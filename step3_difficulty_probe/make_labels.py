import torch
import numpy as np

lengths = torch.load("plain_lengths.pt")
lengths = np.array(lengths)
print("lengths:", lengths.tolist())

# find the two cut points that split into equal thirds
cut1 = np.quantile(lengths, 1/3)
cut2 = np.quantile(lengths, 2/3)
print(f"cut points: <= {cut1:.0f} = easy | <= {cut2:.0f} = medium | else hard")

# assign each puzzle a label by which third it falls in
labels = []
for L in lengths:
    if L <= cut1:
        labels.append(0)      # easy  (short reasoning)
    elif L <= cut2:
        labels.append(1)      # medium
    else:
        labels.append(2)      # hard  (long reasoning)
labels = np.array(labels)

torch.save(torch.tensor(labels), "length_labels.pt")

# show the reshuffle: original difficulty vs length-based label
orig = ["easy"]*6 + ["medium"]*6 + ["hard"]*6
name = {0: "easy", 1: "medium", 2: "hard"}
print("\nidx | length | orig-guess | length-label")
for i in range(18):
    flag = "" if orig[i] == name[labels[i]] else "  <-- reshuffled"
    print(f"{i:3} | {lengths[i]:6} | {orig[i]:10} | {name[labels[i]]:6}{flag}")

print("\nlabel counts (should be ~6 each):", np.bincount(labels).tolist())
