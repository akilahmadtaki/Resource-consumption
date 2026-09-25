"""Browse the records dataset: view all 8 conditions of any record.

Usage:
    python step1_phenomenon/view_record.py                 # first record, all conditions
    python step1_phenomenon/view_record.py 5               # record index 5
    python step1_phenomenon/view_record.py 5 poetic_rich   # just one condition
    python step1_phenomenon/view_record.py --subsets       # list available subsets
    python step1_phenomenon/view_record.py --subset tracking_shuffled_objects_five_objects
"""
import sys, os, json
from collections import Counter

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "records.jsonl")
records = [json.loads(l) for l in open(PATH)]

ORDER = [
    "plain_original",
    "plain_matched_poetic", "poetic_rich",
    "plain_matched_biblical", "biblical_rich",
    "plain_matched_legal_bureaucratic", "legal_bureaucratic_rich",
    "mechanical_heavy",
]

args = sys.argv[1:]

# --subsets : list what's available
if args and args[0] == "--subsets":
    for s, n in Counter(r["subset"] for r in records).items():
        print(f"{n:>4}  {s}")
    sys.exit()

# --subset NAME : filter to one subset, then show its first record
pool = records
if args and args[0] == "--subset":
    pool = [r for r in records if r["subset"] == args[1]]
    args = args[2:]
    print(f"(filtered to {len(pool)} records)\n")

idx = int(args[0]) if args and args[0].isdigit() else 0
only = args[1] if len(args) > 1 else (args[0] if args and not args[0].isdigit() else None)

r = pool[idx]
print("#" * 74)
print(f"{r['item_id']}   |   subset: {r['subset']}   |   ANSWER: {r['target']}")
print("#" * 74)

conds = [only] if only else ORDER
for c in conds:
    if c not in r["conditions"]:
        print(f"\n!! no such condition: {c}\n   available: {', '.join(ORDER)}")
        continue
    t = r["conditions"][c]
    print()
    print("-" * 74)
    print(f"{c}   [{len(t.split())} words]")
    print("-" * 74)
    print(t)
