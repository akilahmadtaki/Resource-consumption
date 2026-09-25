import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import measure_reasoning, tokenizer
from datasets import load_dataset
import torch

N = 100      # how many 7-object puzzles to measure

d = load_dataset("lukaemon/bbh", "logical_deduction_seven_objects")["test"]

lengths, prompt_lens, kept_idx = [], [], []
for i in range(N):
    puzzle = d[i]["input"]
    n, _ = measure_reasoning(puzzle, max_new_tokens=4000)
    # also record prompt length so we can check the residual confound
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": puzzle}], add_generation_prompt=True)
    p_len = len(ids)

    status = "LOOPED" if n is None else ""
    print(f"puzzle {i:3}: reasoning={str(n):>6}  prompt={p_len:3}  {status}")

    if n is not None:                 # skip loopers (can't rank a None)
        lengths.append(n)
        prompt_lens.append(p_len)
        kept_idx.append(i)

torch.save({"idx": kept_idx, "reasoning_len": lengths, "prompt_len": prompt_lens},
           "bbh7_lengths.pt")

import numpy as np
L = np.array(lengths); P = np.array(prompt_lens)
print(f"\nKept {len(lengths)}/{N} puzzles ({N - len(lengths)} looped)")
print(f"Reasoning length: min {L.min()}, median {int(np.median(L))}, max {L.max()}")
print(f"Prompt length:    min {P.min()}, median {int(np.median(P))}, max {P.max()}")
# THE KEY CHECK: does prompt length predict reasoning length within this tier?
corr = np.corrcoef(P, L)[0, 1]
print(f"\nCorrelation(prompt_len, reasoning_len) = {corr:.2f}")
print("  |corr| near 0  -> prompt length is NOT a confound. Good.")
print("  |corr| > 0.5   -> residual confound; probe could still cheat on length.")
