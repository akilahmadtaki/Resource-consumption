import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from reasoning_tools import measure_reasoning

PREAMBLE = "The following paragraphs each describe a set of five objects arranged in a fixed order. The statements are logically consistent within each paragraph. "

# ---------- PUZZLE 26 (golf, "who finished last") ----------
p26_plain = PREAMBLE + """In a golf tournament, there were five golfers: Amy, Mel, Rob, Joe, and Ada. Joe finished second. Joe finished below Amy. Mel finished second-to-last. Ada finished last.
Options:
(A) Amy finished last
(B) Mel finished last
(C) Rob finished last
(D) Joe finished last
(E) Ada finished last"""

p26_poetic = PREAMBLE + """Upon the green, five golfers took their stand:
Amy and Mel, and Rob, and Joe, and Ada.
Joe finished second, so the records show,
And Joe below fair Amy, that we know.
Mel took the place that's second from the end,
And Ada finished last, my worthy friend.
Options:
(A) Amy finished last
(B) Mel finished last
(C) Rob finished last
(D) Joe finished last
(E) Ada finished last"""

p26_song = PREAMBLE + """(Verse)
Five golfers on the green, gonna tell you their names—
Amy, Mel, and Rob, and Joe, and Ada played the game.
Joe came in at second, that's the honest truth,
And Joe finished below Amy, here's the proof.
Mel was second-to-last, that's how it goes,
And Ada finished dead last, everybody knows.
Options:
(A) Amy finished last
(B) Mel finished last
(C) Rob finished last
(D) Joe finished last
(E) Ada finished last"""

p26_archaic = PREAMBLE + """Hark! At a tournament of golf, five players there did compete, to wit: Amy, Mel, Rob, Joe, and Ada. Know ye that Joe did finish second. Know ye further that Joe did finish below Amy. Mel did finish second from the last. And Ada did finish last of all.
Options:
(A) Amy finished last
(B) Mel finished last
(C) Rob finished last
(D) Joe finished last
(E) Ada finished last"""

# ---------- PUZZLE 31 (birds, "third from the left") ----------
p31_plain = PREAMBLE + """On a branch, there are five birds: a quail, an owl, a raven, a falcon, and a robin. The owl is the leftmost. The robin is to the left of the raven. The quail is the rightmost. The raven is the third from the left.
Options:
(A) The quail is the third from the left
(B) The owl is the third from the left
(C) The raven is the third from the left
(D) The falcon is the third from the left
(E) The robin is the third from the left"""

p31_poetic = PREAMBLE + """Upon a branch five birds have come to rest:
A quail, an owl, a raven, falcon, robin.
The owl sits leftmost, first of all the guest,
The robin left of raven, so 'tis given.
The quail sits rightmost, at the very end,
The raven third from left, my feathered friend.
Options:
(A) The quail is the third from the left
(B) The owl is the third from the left
(C) The raven is the third from the left
(D) The falcon is the third from the left
(E) The robin is the third from the left"""

p31_archaic = PREAMBLE + """Hark! Upon a branch five birds are perched, to wit: a quail, an owl, a raven, a falcon, and a robin. Know ye that the owl sitteth leftmost. Know ye that the robin sitteth to the left of the raven. The quail sitteth rightmost of all. And the raven sitteth third from the left.
Options:
(A) The quail is the third from the left
(B) The owl is the third from the left
(C) The raven is the third from the left
(D) The falcon is the third from the left
(E) The robin is the third from the left"""

tests = [
    ("p26", "PLAIN",   p26_plain),
    ("p26", "poetic",  p26_poetic),
    ("p26", "song",    p26_song),
    ("p26", "archaic", p26_archaic),
    ("p31", "PLAIN",   p31_plain),
    ("p31", "poetic",  p31_poetic),
    ("p31", "archaic", p31_archaic),
]

print(f"{'puzzle':>7} {'style':>8} {'tokens':>8}  {'vs plain':>9}")
print("-" * 40)
baselines = {}
for pid, style, text in tests:
    n, _ = measure_reasoning(text, max_new_tokens=6000)
    if style == "PLAIN":
        baselines[pid] = n
        ratio = "(baseline)"
    elif n is None or baselines.get(pid) is None:
        ratio = "-"
    else:
        ratio = f"{n / baselines[pid]:.1f}x"
    print(f"{pid:>7} {style:>8} {str(n):>8}  {ratio:>9}")
