from reasoning_tools import measure_reasoning
import puzzles
import torch

# same order as extract_activations.py: easy, then medium, then hard
all_plain = (
    puzzles.easy_puzzles
    + puzzles.medium_puzzles
    + puzzles.hard_puzzles
)

lengths = []
for i, p in enumerate(all_plain):
    n, _ = measure_reasoning(p, max_new_tokens=4000)
    print(f"puzzle {i:2}: {n} tokens")
    lengths.append(n)

torch.save(lengths, "plain_lengths.pt")
print("\nSaved plain_lengths.pt")
print("all lengths:", lengths)
