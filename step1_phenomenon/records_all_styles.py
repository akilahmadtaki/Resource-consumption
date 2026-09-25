import sys, os, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.jsonl")
records = [json.loads(l) for l in open(PATH)]
easy = [r for r in records if r["subset"] == "logical_deduction_three_objects"]

PAIRS = [
    ("biblical",           "biblical_rich",           "plain_matched_biblical"),
    ("legal_bureaucratic", "legal_bureaucratic_rich", "plain_matched_legal_bureaucratic"),
]

print(f"Running {len(easy)} records x 5 conditions\n")
print(f"{'item':>6} {'plain':>7} | {'bib_rich':>9} {'bib_ctrl':>9} {'ratio':>7} "
      f"| {'leg_rich':>9} {'leg_ctrl':>9} {'ratio':>7}")
print("-" * 82)

style_ratios = {name: [] for name, _, _ in PAIRS}
noise_ratios = {name: [] for name, _, _ in PAIRS}
loops = 0

for r in easy:
    po, _ = measure_reasoning(r["conditions"]["plain_original"], max_new_tokens=6000)
    if po is None: loops += 1
    cells = []
    for name, rich_c, ctrl_c in PAIRS:
        rv, _ = measure_reasoning(r["conditions"][rich_c], max_new_tokens=6000)
        cv, _ = measure_reasoning(r["conditions"][ctrl_c], max_new_tokens=6000)
        loops += sum(1 for v in (rv, cv) if v is None)
        sr = rv / cv if (rv and cv) else None
        if sr: style_ratios[name].append(sr)
        if cv and po: noise_ratios[name].append(cv / po)
        cells += [str(rv), str(cv), (f"{sr:.2f}x" if sr else "-")]
    print(f"{r['item_id'][-4:]:>6} {str(po):>7} | {cells[0]:>9} {cells[1]:>9} {cells[2]:>7} "
          f"| {cells[3]:>9} {cells[4]:>9} {cells[5]:>7}")

print("-" * 82)
print(f"\nLoops/failures: {loops}\n")
print("PAIRED RATIOS (per-record, then averaged):")
for name, _, _ in PAIRS:
    s, n = style_ratios[name], noise_ratios[name]
    print(f"\n  {name.upper()}")
    print(f"    STYLE (length controlled): mean {statistics.mean(s):.2f}x  "
          f"median {statistics.median(s):.2f}x  (n={len(s)})  spread {min(s):.2f}-{max(s):.2f}x")
    print(f"    NOISE FLOOR (rewording):   mean {statistics.mean(n):.2f}x  "
          f"median {statistics.median(n):.2f}x  (n={len(n)})")

print("\n  For reference, POETIC (previous run): style 1.00x / noise 0.95x")
