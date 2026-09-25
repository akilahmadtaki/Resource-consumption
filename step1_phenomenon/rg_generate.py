import reasoning_gym, json, os

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "rg_puzzles.jsonl")

rows = []
for size in [4, 6, 8]:
    data = reasoning_gym.create_dataset(
        "family_relationships",
        size=5, seed=123,
        min_family_size=size, max_family_size=size,
    )
    for i, item in enumerate(data):
        rows.append({
            "task": "family_relationships",
            "family_size": size,
            "idx": i,
            "question": item["question"],
            "answer": item["answer"],
            "words": len(item["question"].split()),
        })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"wrote {len(rows)} puzzles -> {OUT}")
for size in [4, 6, 8]:
    ws = [r["words"] for r in rows if r["family_size"] == size]
    print(f"  size {size}: {len(ws)} puzzles, words {min(ws)}-{max(ws)}")
