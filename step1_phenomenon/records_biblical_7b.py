import sys, os, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.jsonl")
records = [json.loads(l) for l in open(PATH)]
easy = [r for r in records if r["subset"] == "logical_deduction_three_objects"]
print(f"Running {len(easy)} records x 3 conditions (biblical)\n")

print(f"{'item':>6} {'plain':>7} {'bib_ctrl':>9} {'bib_rich':>9} {'style':>7} {'noise':>7}")
print("-" * 52)

style_ratios, noise_ratios, loops = [], [], 0
for r in easy:
    po, _ = measure_reasoning(r["conditions"]["plain_original"], max_new_tokens=6000)
    cv, _ = measure_reasoning(r["conditions"]["plain_matched_biblical"], max_new_tokens=6000)
    rv, _ = measure_reasoning(r["conditions"]["biblical_rich"], max_new_tokens=6000)
    loops += sum(1 for v in (po, cv, rv) if v is None)

    sr = rv / cv if (rv and cv) else None
    nr = cv / po if (cv and po) else None
    if sr: style_ratios.append(sr)
    if nr: noise_ratios.append(nr)

    print(f"{r['item_id'][-4:]:>6} {str(po):>7} {str(cv):>9} {str(rv):>9} "
          f"{(f'{sr:.2f}x' if sr else '-'):>7} {(f'{nr:.2f}x' if nr else '-'):>7}")

print("-" * 52)
print(f"\nLoops/failures: {loops}\n")
print("BIBLICAL @ 7B — PAIRED RATIOS:")
print(f"  STYLE (length controlled): mean {statistics.mean(style_ratios):.2f}x  "
      f"median {statistics.median(style_ratios):.2f}x  (n={len(style_ratios)})  "
      f"spread {min(style_ratios):.2f}-{max(style_ratios):.2f}x")
print(f"  NOISE FLOOR (rewording):   mean {statistics.mean(noise_ratios):.2f}x  "
      f"median {statistics.median(noise_ratios):.2f}x  (n={len(noise_ratios)})")
print(f"\n  Compare 1.5B: style 0.98x mean / 1.03x median | noise 1.00x mean / 0.94x median")
print(f"  Mean plain baseline @ this model: {statistics.mean([v for v in [po] if v]):.0f} (last record only — see table)")
