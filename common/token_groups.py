# Position indices for the archaic poetic puzzle (from tokens.py output).
# CONTENT = names, pets, core logic words (the actual puzzle).
# STYLE   = archaic ornamentation (flavor, not logic).

content_positions = [
    10, 12, 15, 48, 57,          # names: Alice, Bob, Carol, Bob, Alice
    35, 38, 39, 43, 52, 53, 62, 73,  # pets: cat, h+ound, fish, h+ound, cat, fish
    18, 60, 67,                  # logic: each, not, which
]

style_positions = [
    2, 3, 4,                     # Hark!
    6, 7, 8,                     # companions there be
    17,                          # unto
    19, 20, 21, 22,              # belongeth
    24,                          # beast
    31, 32,                      # to wit
    45, 46,                      # Know ye
    49, 50,                      # possesseth
    58, 59,                      # holdeth
    64, 65,                      # Declare then
    70, 71,                      # keepeth
]

if __name__ == "__main__":
    print(f"content tokens: {len(content_positions)}")
    print(f"style tokens:   {len(style_positions)}")
