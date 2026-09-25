import sys, os, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rg_puzzles.jsonl")
rows = [json.loads(l) for l in open(PATH)]

print(f"{'size':>5} {'words':>6} {'tokens':>8}")
print("-" * 22)
by_size = {}
for r in rows:
    n, _ = measure_reasoning(r["question"], max_new_tokens=4000)
    by_size.setdefault(r["family_size"], []).append(n)
    print(f"{r['family_size']:>5} {r['words']:>6} {str(n):>8}")

print("-" * 22)
print("\nMEAN PLAIN BASELINE:")
for size, vals in sorted(by_size.items()):
    ok = [v for v in vals if v is not None]
    if ok:
        print(f"  size {size}: {statistics.mean(ok):.0f} tokens  (n={len(ok)}, "
              f"{len(vals)-len(ok)} loops)")
print("\nReference: hand-written easy = 66 | BBH 3-object = ~480")
