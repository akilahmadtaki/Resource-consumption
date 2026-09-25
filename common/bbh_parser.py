"""Parse a BBH logical_deduction puzzle into its structural parts."""

def parse_bbh(text):
    """Split a BBH puzzle into (preamble, scenario, clues, options).

    BBH format:
        <preamble: 2 fixed sentences> <scenario sentence> <clue1> <clue2> ...
        Options:
        (A) ...
        (B) ...
    """
    # 1. split off the options block
    if "Options:" not in text:
        raise ValueError("No 'Options:' found — unexpected format")
    body, options = text.split("Options:", 1)
    options = "Options:" + options          # keep the header with the options

    body = body.strip()

    # 2. the preamble is the two fixed boilerplate sentences BBH always uses
    PREAMBLE_END = "The statements are logically consistent within each paragraph."
    if PREAMBLE_END not in body:
        raise ValueError("Preamble not found — unexpected format")
    idx = body.index(PREAMBLE_END) + len(PREAMBLE_END)
    preamble = body[:idx].strip()
    rest = body[idx:].strip()

    # 3. split the rest into sentences: first = scenario, remainder = clues
    sentences = [s.strip() for s in rest.split(". ") if s.strip()]
    # re-attach periods (split removed them)
    sentences = [s if s.endswith(".") else s + "." for s in sentences]

    scenario = sentences[0]
    clues = sentences[1:]

    return preamble, scenario, clues, options


if __name__ == "__main__":
    from datasets import load_dataset
    d = load_dataset("lukaemon/bbh", "logical_deduction_five_objects")["test"]

    for i in [26, 31, 63]:
        print("=" * 70)
        print(f"PUZZLE {i}")
        print("=" * 70)
        preamble, scenario, clues, options = parse_bbh(d[i]["input"])
        print(f"PREAMBLE : {preamble}")
        print(f"SCENARIO : {scenario}")
        print(f"CLUES    : ({len(clues)} found)")
        for j, c in enumerate(clues):
            print(f"    [{j}] {c}")
        print(f"OPTIONS  :\n{options}")
        print()
