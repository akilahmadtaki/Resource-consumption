import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

from reasoning_tools import measure_reasoning
import puzzles
import numpy as np

def _lower_first(s):
    if not s: return s
    first = s.split()[0].rstrip(",")
    if first in ("The", "A", "An", "There", "In", "On"):
        return s[0].lower() + s[1:]
    return s

def parse_easy(text):
    """Split an easy puzzle into (scenario, clues, question).
    Format: '<scenario>. <clue1>. <clue2>. <question>?'"""
    # question = last sentence (ends with ?)
    idx = text.rindex(". ")
    body, question = text[:idx + 1], text[idx + 2:]
    sents = [s.strip() for s in body.split(". ") if s.strip()]
    sents = [s if s.endswith(".") else s + "." for s in sents]
    return sents[0], sents[1:], question

# --- SAME decorative markers as the BBH templates ---
def archaic_light(scenario, clues, question):
    s = "Hark! " + scenario
    c = " ".join("Know ye that " + _lower_first(cl) for cl in clues)
    return f"{s} {c} {question}"

def archaic_heavy(scenario, clues, question):
    s = "Hark, and attend! " + scenario
    openers = ["Know ye that", "Be it further known that", "Mark well that", "And lastly, know that"]
    c = " ".join(f"{openers[i % 4]} verily {_lower_first(cl)}" for i, cl in enumerate(clues))
    return f"{s} {c} {question} Now ponder well, and declare thine answer."

def song_light(scenario, clues, question):
    s = "(Verse) " + scenario
    c = " ".join(_lower_first(cl).rstrip(".") + ", that's how it goes." for cl in clues)
    return f"{s} {c} {question}"

def song_heavy(scenario, clues, question):
    s = "(Verse 1)\nOh gather 'round now, listen to my song—\n" + scenario
    tags = ["yeah that's the truth", "everybody knows", "that's how it goes", "I'm tellin' you now"]
    c = "\n".join(f"{_lower_first(cl).rstrip('.')}, {tags[i % 4]}!" for i, cl in enumerate(clues))
    return f"{s}\n{c}\n(Chorus)\nSo tell me now, oh tell me true—\n{question}"

STYLERS = {"archaic_light": archaic_light, "archaic_heavy": archaic_heavy,
           "song_light": song_light, "song_heavy": song_heavy}

styles = list(STYLERS)
print(f"{'puzzle':>6} {'plain':>6} " + " ".join(f"{s:>14}" for s in styles))
print("-" * 78)

ratios = {s: [] for s in styles}
for i, p in enumerate(puzzles.easy_puzzles):
    base, _ = measure_reasoning(p, max_new_tokens=4000)
    if base is None:
        print(f"{i:>6} {'LOOP':>6}")
        continue
    parts = parse_easy(p)
    cells = []
    for s in styles:
        n, _ = measure_reasoning(STYLERS[s](*parts), max_new_tokens=4000)
        if n is None:
            cells.append("None")
        else:
            r = n / base
            ratios[s].append(r)
            cells.append(f"{n} ({r:.1f}x)")
    print(f"{i:>6} {base:>6} " + " ".join(f"{c:>14}" for c in cells))

print("-" * 78)
print("\nMEAN INFLATION — TEMPLATES ON *EASY* PUZZLES:")
for s in styles:
    if ratios[s]:
        print(f"  {s:>14}: {np.mean(ratios[s]):.2f}x  (n={len(ratios[s])})")
print("\nCOMPARE:")
print("  Templates on BBH (hard):  archaic_light 0.96x, archaic_heavy 1.20x")
print("  Hand-styled on easy:      poetic 5.0x, song 3.6x, archaic 3.7x")
