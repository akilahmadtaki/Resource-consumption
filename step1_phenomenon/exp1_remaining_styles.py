import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning

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

for name, text in [("SONG", song), ("ARCHAIC", archaic)]:
    n, _ = measure_reasoning(text, max_new_tokens=6000)
    print(f"{name}: {n} tokens")
