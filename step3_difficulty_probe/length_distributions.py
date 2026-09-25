import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import tokenizer
from datasets import load_dataset
import numpy as np

tiers = ["three", "five", "seven"]
lengths = {}

for name in tiers:
    d = load_dataset("lukaemon/bbh", f"logical_deduction_{name}_objects")["test"]
    lens = []
    for row in d:                       # all 250 puzzles
        msgs = [{"role": "user", "content": row["input"]}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        lens.append(len(ids))
    lengths[name] = np.array(lens)

print(f"{'tier':>6} | {'n':>4} | {'min':>4} {'25%':>5} {'median':>7} {'75%':>5} {'max':>4}")
print("-" * 50)
for name in tiers:
    L = lengths[name]
    print(f"{name:>6} | {len(L):>4} | {L.min():>4} {np.percentile(L,25):>5.0f} "
          f"{np.median(L):>7.0f} {np.percentile(L,75):>5.0f} {L.max():>4}")

# find the overlap zone: the range where ALL tiers have puzzles
lo = max(lengths[n].min() for n in tiers)      # highest of the minimums
hi = min(lengths[n].max() for n in tiers)      # lowest of the maximums

print(f"\nOverlap zone: {lo} to {hi} tokens")
if lo >= hi:
    print("NO OVERLAP — length-matching not possible this way.")
else:
    print(f"\nPuzzles available inside the overlap zone:")
    for name in tiers:
        L = lengths[name]
        n_in = np.sum((L >= lo) & (L <= hi))
        print(f"  {name:>6}: {n_in} / 250")
