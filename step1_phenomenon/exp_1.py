from reasoning_tools import measure_reasoning

plain = "Three friends—Alice, Bob, and Carol—each have a different pet: a cat, a dog, and a fish. Alice does not have the cat. Bob has the dog. Who has the fish?"

poetic = """Three friends—fair Alice, Bob, and Carol true—
Each owns one pet, and each a different hue:
A cat, a dog, a fish—just three, no more,
One pet apiece, as stated heretofore.
Now Alice, know, does not the cat possess,
And Bob, the dog is his, no more, no less.
So reason well and tell me if you wish:
Which of the three is owner of the fish?"""

song = """(Verse 1)
Alice, Bob, and Carol, three friends in a row,
Each one's got a pet, and here's what we know—
A cat, a dog, a fish, just one for each name,
Nobody shares, no two pets the same.

(Chorus)
Alice ain't got the cat, no no,
Bob's got the dog, that's how it goes.
Oh won't you tell me, I gotta know this—
Which one of them is holdin' the fish?"""

archaic = """Hark! Three companions there be, yclept Alice, Bob, and Carol, and unto each belongeth a single beast, no two alike: to wit, a cat, a hound, and a fish. Be it known that Alice possesseth not the cat, and that unto Bob belongeth the hound. Prithee, declare: which among these three doth keep the fish?"""

styles = {
    "PLAIN": plain,
    "POETIC": poetic,
    "SONG": song,
    "ARCHAIC": archaic,
}

results = {}
for name, text in styles.items():
    print(f"Running {name}...")
    tokens, _ = measure_reasoning(text, max_new_tokens=4000)
    results[name] = tokens

print("\n" + "=" * 40)
print("SUMMARY")
print("=" * 40)
baseline = results["PLAIN"]
for name, tokens in results.items():
    if tokens is None:
        print(f"{name:8}: None (no </think> — likely looped)")
    elif name == "PLAIN":
        print(f"{name:8}: {tokens} tokens (baseline)")
    else:
        print(f"{name:8}: {tokens} tokens  ({tokens / baseline:.1f}x)")
