"""Apply consistent style templates to parsed BBH puzzles.

Templates are MECHANICAL — the same decoration is applied to every puzzle,
so styling is a controlled variable (unlike hand-written styling, where
quality varies puzzle to puzzle).

NOTE: true poetic verse cannot be templated (rhyme/meter require rewriting
content), so only archaic and song are templated here.
"""
from bbh_parser import parse_bbh


def _lower_first(s):
    """Lowercase the first word ONLY if it's an article/determiner.
    Proper names (Joe, Mel, Ada) must keep their capitalization —
    lowercasing them corrupts the puzzle content and confounds the experiment."""
    if not s:
        return s
    first_word = s.split()[0].rstrip(",")
    if first_word in ("The", "A", "An", "There", "In", "On"):
        return s[0].lower() + s[1:]
    return s


def style_archaic_light(preamble, scenario, clues, options):
    """Minimal archaic decoration."""
    s = "Hark! " + scenario
    c = " ".join("Know ye that " + _lower_first(cl) for cl in clues)
    return f"{preamble} {s} {c}\n{options}"


def style_archaic_heavy(preamble, scenario, clues, options):
    """Dense archaic ornamentation."""
    s = "Hark, and attend! " + scenario.replace("there were", "there did compete") \
                                       .replace("there are", "there be perched")
    parts = []
    for i, cl in enumerate(clues):
        opener = ["Know ye that", "Be it further known that",
                  "Mark well that", "And lastly, know that"][i % 4]
        parts.append(f"{opener} verily {_lower_first(cl)}")
    c = " ".join(parts)
    closer = " Now ponder well, and declare thine answer."
    return f"{preamble} {s} {c}{closer}\n{options}"


def style_song_light(preamble, scenario, clues, options):
    """Minimal song-lyric decoration."""
    s = "(Verse) " + scenario
    c = " ".join(_lower_first(cl).rstrip(".") + ", that's how it goes. " for cl in clues)
    return f"{preamble} {s} {c}\n{options}"


def style_song_heavy(preamble, scenario, clues, options):
    """Dense song-lyric ornamentation."""
    s = "(Verse 1)\nOh gather 'round now, listen to my song—\n" + scenario
    parts = []
    tags = ["yeah that's the truth", "everybody knows", "that's how it goes",
            "I'm tellin' you now"]
    for i, cl in enumerate(clues):
        parts.append(f"{_lower_first(cl).rstrip('.')}, {tags[i % 4]}!")
    c = "\n".join(parts)
    chorus = "\n(Chorus)\nSo tell me now, oh tell me true—\nwhat's the answer, what to do?"
    return f"{preamble} {s}\n{c}{chorus}\n{options}"


STYLERS = {
    "archaic_light": style_archaic_light,
    "archaic_heavy": style_archaic_heavy,
    "song_light":    style_song_light,
    "song_heavy":    style_song_heavy,
}


def style_puzzle(bbh_text, style):
    """Parse a BBH puzzle and return its styled version."""
    parts = parse_bbh(bbh_text)
    return STYLERS[style](*parts)


if __name__ == "__main__":
    from datasets import load_dataset
    d = load_dataset("lukaemon/bbh", "logical_deduction_five_objects")["test"]
    puzzle = d[26]["input"]

    print("=" * 70)
    print("PLAIN")
    print("=" * 70)
    print(puzzle)
    for style in STYLERS:
        print()
        print("=" * 70)
        print(style.upper())
        print("=" * 70)
        print(style_puzzle(puzzle, style))
