import reasoning_gym

for size in [4, 6, 8]:
    data = reasoning_gym.create_dataset(
        "family_relationships",
        size=2,
        seed=42,
        min_family_size=size,
        max_family_size=size,
    )
    print("=" * 70)
    print(f"family_relationships | family_size = {size}")
    print("=" * 70)
    for item in data:
        q = item["question"]
        print(f"[{len(q.split())} words]")
        print(q)
        print("ANSWER:", item["answer"])
        print()
