import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import measure_reasoning
from datasets import load_dataset

for n in ["three", "five", "seven"]:
    subset = f"logical_deduction_{n}_objects"
    d = load_dataset("lukaemon/bbh", subset)["test"]
    puzzle = d[0]["input"]

    print("=" * 70)
    print(f"{subset}")
    print("=" * 70)
    print("PUZZLE:")
    print(puzzle)
    print()
    print("Running through the model...")
    tokens, think = measure_reasoning(puzzle, max_new_tokens=4000)
    print(f"\nReasoning length: {tokens} tokens")
    if think:
        print(f"\nFirst 300 chars of reasoning:\n{think[:300]}...")
    else:
        print("NO </think> found — looped or truncated!")
    print()

