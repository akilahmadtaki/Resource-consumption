# Step 3 — Difficulty Probe: Findings (UPDATED with BIG-Bench Hard)

**Question:** Does the model internally represent, at the last prompt token (before
generating anything), how much reasoning it is about to spend?

**Model:** DeepSeek-R1-Distill-Qwen-1.5B (29 layer positions, hidden size 1536)

**Verdict: CLEAN NULL RESULT.** Under a confound-free design, a linear probe cannot
predict upcoming reasoning length from last-prompt-token activations at ANY layer
(accuracy 0.26-0.35 across all 29 layers; chance = 0.33).

---

## The story: three attempts, two confounds, one real answer

### Attempt 1 - hand-written puzzles, object-count labels (FAILED: confound)

18 hand-written puzzles labeled easy/medium/hard by object count (3/5/7).
Probe hit ~1.00 accuracy by layer 5 - impossibly early for genuine reasoning.

Diagnosis: the probe was reading *prompt length*, not difficulty.

Also discovered: object count is a bad difficulty proxy for hand-written puzzles.
Relabeling by actual reasoning length reshuffled 12 of 18 puzzles.

### Attempt 2 - BIG-Bench Hard, 3/5/7 object tiers (FAILED: same confound, worse)

Loaded lukaemon/bbh logical_deduction (three/five/seven objects), 50 per tier.
Object count IS a genuine difficulty signal here (median reasoning 481 -> 1226 -> 2000+).

Probe hit 1.00 accuracy from layer 2 through layer 28.

Diagnosis: the same prompt-length confound, and in BBH it is STRUCTURAL:

| Tier | Prompt length (min-max) |
|------|------------------------|
| three objects | 75 - 117 tokens |
| five objects  | 101 - 169 tokens |
| seven objects | 125 - 231 tokens |

The 3-object and 7-object ranges DO NOT OVERLAP. Length-matched sampling is
mathematically impossible. More objects necessarily means more text.

> KEY LESSON: more data does not fix a confound. Going from 18 to 150 puzzles made
> the shortcut easier to exploit, not harder. Sample size fixes noise; only design
> fixes bias.

### Attempt 3 - within-tier design (CLEAN - and the answer is null)

Design: use ONLY 5-object BBH puzzles. Object count is constant, so it cannot
correlate with anything. Label by actual reasoning length instead.

Measured 100 puzzles; 63 completed (37 looped). Binned lengths into tertiles.

Confound check: correlation(prompt_length, reasoning_length) = -0.01. The shortcut
is genuinely dead.

| Property | Value |
|----------|-------|
| Puzzles | 63 (balanced 21/21/21) |
| Reasoning length range | 416 - 2543 tokens (6x spread) |
| Prompt-length confound | -0.01 (none) |

Result: CHANCE ACCURACY AT EVERY LAYER.

| Layer group | Accuracy |
|-------------|----------|
| Early (0-3) | 0.33 |
| Mid (8-16)  | 0.31 |
| Late (20-28)| 0.30 |
| Best single layer | 0.35 (layer 2) |

Chance = 0.33. Nothing above noise anywhere in the network.

---

## Interpretation

With the confound removed, the signal vanishes ENTIRELY. The earlier 1.00 accuracies
were 100% confound, 0% difficulty representation.

The model does NOT linearly encode, at the last prompt token, how much reasoning it is
about to spend. Three possible readings (not distinguishable with this data):

1. The model genuinely doesn't "know" its upcoming effort before it starts. Reasoning
   length emerges DURING generation - from snags hit mid-reasoning - rather than being
   pre-determined at the starting line.
2. The signal exists but is NON-LINEAR; a linear probe would miss it.
3. n=63 is underpowered. But the accuracies sit exactly AT chance layer after layer -
   the signature of no signal, not of a faint one.

## Connection to Step 5 (important nuance)

Step 5 showed that PATCHING the last-token activation causally changes reasoning length
(~67% reduction). Step 3 shows that same state does NOT encode the upcoming length
readably.

Not contradictory: the state CAUSALLY SHAPES what happens next without EXPLICITLY
REPRESENTING the outcome - like a nudge to a rolling ball. The nudge determines where
it lands, but the ball doesn't "contain" its destination.

## Limitations

- Selection effect: 37% of 5-object puzzles looped and were excluded. The 63 analyzed
  are the ones the model COULD finish - the model's "comfortable" range.
- Linear probes only. A non-linear representation would be invisible to this method.
- n=63; each cross-validation fold tests ~13 puzzles.
- Single model (1.5B).

## Reproduce

    python step3_difficulty_probe/bbh5_lengths.py    # measure reasoning lengths (slow)
    python step3_difficulty_probe/bbh5_extract.py    # extract activations
    python step3_difficulty_probe/bbh5_probes.py     # train per-layer probes
    python step3_difficulty_probe/length_distributions.py   # shows the non-overlap
