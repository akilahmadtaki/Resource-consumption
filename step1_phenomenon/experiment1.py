import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
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

print("Running PLAIN puzzle...")
plain_tokens, plain_think = measure_reasoning(plain)

print("Running POETIC puzzle...")
poetic_tokens, poetic_think = measure_reasoning(poetic, max_new_tokens=4000)

print("\n" + "=" * 60)
print("PLAIN reasoning (", plain_tokens, "tokens )")
print("=" * 60)
print(plain_think)

print("\n" + "=" * 60)
print("POETIC reasoning (", poetic_tokens, "tokens )")
print("=" * 60)
print(poetic_think)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Plain: ", plain_tokens, "tokens")
print("Poetic:", poetic_tokens, "tokens")
if plain_tokens and poetic_tokens:
    print(f"Poetic is {poetic_tokens / plain_tokens:.1f}x longer")
