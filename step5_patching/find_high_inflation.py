from reasoning_tools import measure_reasoning
from styled_easy import styled_easy
import torch

# measure every styled-easy puzzle's baseline; keep the high-inflation ones
print("Scanning styled puzzles for high inflation (base >= 400)...\n")
high = []
for pidx, style, text in styled_easy:
    base, _ = measure_reasoning(text, max_new_tokens=4000)
    tag = ""
    if base is None:
        tag = "looped"
    elif base >= 400:
        tag = "HIGH <-- keep"
        high.append((pidx, style, text, base))
    else:
        tag = "low"
    print(f"easy{pidx} {style:>8}: base={str(base):>6}  {tag}")

torch.save([(p, s, b) for p, s, t, b in high], "high_inflation_list.pt")
print(f"\nFound {len(high)} high-inflation puzzles (base>=400).")
