# Step 3 — Difficulty Probe: Findings

**Question:** Does stylistic framing make the model *internally represent* an easy
puzzle as harder — before it generates any tokens? (Tested via a linear probe on
last-prompt-token activations.)

**Model:** DeepSeek-R1-Distill-Qwen-1.5B (28 layers, hidden size 1536).

---

## What we did

1. Built 18 plain logic puzzles (6 easy / 6 medium / 7 hard by object count: 3/5/7 objects).
2. Extracted last-prompt-token activations at all 29 layer positions.
3. Trained a linear probe (logistic regression) per layer to predict difficulty.
4. Tested the probe on styled-easy puzzles (poetic / song / archaic) to see if
   style pushes them into a harder class.

## Key methodological findings (these are the real value of Step 3)

- **Object-count is a poor difficulty label.** When we relabeled puzzles by actual
  reasoning length, **12 of 18 puzzles reshuffled** difficulty tier. Some 3-object
  "easy" puzzles took 750+ reasoning tokens; some 7-object "hard" puzzles finished
  in ~80. Object-count and the model's actual effort disagree two-thirds of the time.

- **Probes exploit confounds.** With object-count labels, the probe hit ~1.00
  accuracy by **layer 2** — far too early for genuine reasoning. This was the probe
  reading surface prompt length (more objects = longer prompt), not difficulty.

- **Length-based labels gave a healthier signal.** Accuracy then built *gradually*
  with depth, peaking ~0.83 around **layer 10** — the fingerprint of a real feature
  that forms during processing, not a surface shortcut.

- **Baseline controls are essential.** An early "positive" result (styled-easy called
  harder) collapsed once we measured the probe's error on *plain-easy* controls:
  it misclassified 4/6 plain-easy puzzles as harder-than-easy, because two
  length-outlier "easy" puzzles had poisoned the class boundary.

## The main result: INCONCLUSIVE (instrument-limited)

After cleaning the dataset (dropping 4 puzzles whose object-count and length labels
clashed by 2 tiers), the plain-easy baseline dropped to a clean 0/6, and styled-easy
puzzles were inflated ~6/6 — which *looked* like a strong confirmation.

**But a style-confound sanity check broke it.** Testing whether the probe could
distinguish styled-EASY from styled-HARD showed the probe collapses **almost all
styled puzzles toward "medium"**, regardless of true difficulty. This is consistent
with the probe being confused by out-of-distribution input (styled text is unlike
any plain training puzzle) and retreating to the middle class — NOT with genuine
difficulty inflation.

The styled-hard test set was only **n=3**, far too small to interpret; percentages
swung 33 points from a single puzzle flipping.

**Conclusion:** We cannot claim that style makes easy puzzles look hard to the model.
The probe is too data-starved (14 train / 3 styled-hard test) and too easily confused
by out-of-distribution styled inputs to answer the question reliably.

## What would settle it (future work)

- Expand to ~15-20 puzzles per difficulty class (≥45 total) with length and
  object-count in agreement, so the difficulty label is unambiguous.
- Include styled versions across all difficulty levels (not just easy) so the
  easy-vs-hard-under-style distinction can be measured with adequate n.
- Consider training the probe on a mix of plain AND styled puzzles so styled inputs
  are no longer out-of-distribution.

## Status

Step 3 banked as **partially supported / instrument-limited**. The "where does style
enter" question is pursued from a more direct angle in Step 4 (attention heads),
which does not depend on this probe.