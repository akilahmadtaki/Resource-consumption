# manually enter the results we just got to analyze the pattern
data = [
    ("easy0 poetic",  1209, 115),
    ("easy0 archaic",  416, 123),
    ("easy1 poetic",   357, 302),
    ("easy1 archaic",  306, 320),
    ("easy2 archaic",   97, 696),
]
print(f"{'pair':>15} {'base':>6} {'patched':>8} {'reduction':>10}")
print("-" * 45)
big, small = [], []
for name, base, patched in data:
    red = 100*(base-patched)/base
    print(f"{name:>15} {base:>6} {patched:>8} {red:>9.0f}%")
    (big if base >= 400 else small).append(red)

import statistics
print("-" * 45)
print(f"HIGH-inflation puzzles (base>=400): mean reduction {statistics.mean(big):.0f}%  {[f'{r:.0f}%' for r in big]}")
print(f"LOW-inflation puzzles  (base<400):  mean reduction {statistics.mean(small):.0f}%  {[f'{r:.0f}%' for r in small]}")
