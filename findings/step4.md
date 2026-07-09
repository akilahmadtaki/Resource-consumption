# Step 4 — Attention Heads: Findings

**Question:** Are there specific attention heads that fixate on *style* tokens
(ornamentation) rather than *content* tokens (puzzle logic), and might drive the
extra reasoning?

**Model:** DeepSeek-R1-Distill-Qwen-1.5B, eager attention (28 layers x 12 heads = 336 heads).

---

## Method notes (hard-won)

- **`output_attentions=True` requires `attn_implementation="eager"`.** The default
  SDPA implementation discards attention weights (returns None). Eager is slower/
  heavier but exposes them — fine for single-puzzle analysis.

- **Attention sinks dominate raw last-token attention.** The final token dumps
  ~77% of its attention onto sequence-start tokens (begin-of-sequence, <User>) and
  scaffold tokens (<think>, newline). Measuring "last-token attention to style vs
  content" was drowned out by these sinks and produced meaningless million-fold ratios.

- **A second sink hides in the first content token.** Even after excluding positions
  0-1, one early token ('H' of "Hark") absorbed 96% of a head's puzzle-token attention.
  Lesson: exclude seq-start AND first-content anchors.

- **Fix that worked ("Fix B"):** average attention across ALL query positions (not just
  the last token), restricted to real puzzle tokens, sinks excluded. This dilutes
  anchors and reveals genuine head behavior. Ratios became sane (2-10x).

- **Mandatory eyeball-check:** always inspect what the top head ACTUALLY attends to
  before trusting a ratio. This caught every artifact.

## Controls (this is what makes the finding trustworthy)

- **Plain comparison:** a genuine style head should attend to content MORE when style
  is removed (it refocuses). Tested candidate heads on plain vs styled; kept only heads
  whose content-attention rose in plain. This eliminated positional artifacts (e.g.
  heads 27,0 and 17,11 attended the same way regardless of style).

- **Second-style confirmation:** re-ran the plain-vs-styled test with a POETIC puzzle
  (vs the original ARCHAIC). Kept only heads that were style-specific across BOTH styles.

## Confirmed result

**Robust style heads (style-specific across archaic AND poetic, sizable effects):**

| Head | content-attn plain vs styled (archaic) | vs styled (poetic) |
|------|----------------------------------------|--------------------|
| (21, 1) | 0.0068 vs 0.0032 | 0.0068 vs 0.0038 |
| (21, 2) | 0.0040 vs 0.0025 | 0.0040 vs 0.0022 |
| (23, 2) | 0.0013 vs 0.0007 | 0.0013 vs 0.0007 |
| (26, 4) | 0.0017 vs 0.0012 | 0.0017 vs 0.0013 |

All cluster in **layers 21-26**. When style is present, these heads divert attention
from puzzle content onto style ornamentation; remove the style and they refocus on content.

**Downgraded:**
- (25, 8): style-specific for archaic but flat for poetic — possible archaic-specialist,
  does not generalize.
- (27, 0): marginal/noise — absolute attention values (~0.0010) too small to trust verdict.

## Convergence with Step 2

The confirmed style heads (layers 21-26) sit inside the Step-2 logit-lens divergence
band (~21-27), where plain and styled runs diverged in their next-token predictions.
**Two independent methods point at the same region.** This is the strongest result of Step 4.

## Important caveat

**Attention diversion is CORRELATION, not proven causation.** These heads *look away*
from content toward style, but whether that diversion *causes* the longer reasoning is
what Step 5 (activation patching) must test. Step 4 nominates suspects; Step 5 is the trial.

## Step 5 patching targets

Primary: heads (21,1), (21,2), (23,2), (26,4).
Fallback if head-level patching is too weak (effect was somewhat diffuse across many
heads): patch whole layers in the 21-26 band.