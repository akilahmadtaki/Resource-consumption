# Style-Induced Reasoning Inflation: A Mechanistic Investigation

**Model:** DeepSeek-R1-Distill-Qwen-1.5B (28 layers, 12 heads/layer, hidden size 1536)
**Question:** Does dressing a simple logic puzzle in stylistic framing (poetry, song
lyrics, archaic prose) make the model produce longer chain-of-thought — and if so,
*where* inside the model does this happen, and *what causes* it?

---

## Executive summary

Stylistic framing reliably inflates reasoning length ~3.6–5x on this model without
changing the puzzle's logic. The inflation signal is carried in a **distributed** way
across the mid-to-late-layer residual stream (roughly layers 12–27). Causal
intervention (activation patching) on the last-token residual state removes ~67% of
the inflation on average. Crucially, patching the specific "style-attending" attention
heads alone does NOT reproduce this effect — so the overthinking is driven by the
broader integrated state, not by a small localized head circuit.

---

## Step 1 — The phenomenon (CONFIRMED, robust)

Measured `<think>...</think>` token length for one plain puzzle vs. three styled
versions of the SAME puzzle:

| Style   | tokens | vs plain |
|---------|--------|----------|
| Plain   | 66     | —        |
| Poetic  | 329    | 5.0x     |
| Song    | 239    | 3.6x     |
| Archaic | 246    | 3.7x     |

All three styles inflate. Reading the traces showed the LOGIC is identical across
plain and styled — style inflates *quantity* (preamble, over-formalization,
self-verification, restatement), not the reasoning *structure*.

**Side finding:** ambiguous styled phrasing can push the model into non-terminating
loops (never closes `</think>`). Loop-free wording is required for clean measurement.

## Step 2 — Localization via logit lens (CONFIRMED)

Decoded the residual stream at every layer (RMSNorm + lm_head) at the last prompt
token, plain vs. styled. Plain commits to its opening token ("First") by ~layer 21;
styled versions converge ~4 layers later (~layer 25), often onto a different opening
register ("Okay", the verbose-preamble mode). Divergence band: **layers ~21–27.**

Recurring motif: a cross-lingual waypoint (Chinese "首先" = "First") appears one layer
before the English token — meaning is represented before surface form.

## Step 3 — Difficulty probe (INCONCLUSIVE, instrument-limited)

Trained linear probes on last-token activations to predict puzzle difficulty, then
tested whether styled-easy puzzles get read as "hard."

Key lessons (the real value here):
- Object-count is a poor difficulty label (12/18 puzzles reshuffled when relabeled by
  actual reasoning length).
- Probes exploit confounds (early-layer ~1.0 accuracy = reading prompt length, not
  difficulty).
- Baseline controls are essential; an apparent positive result collapsed once the
  plain-easy false-positive rate was measured.
- A style-confound check showed the probe collapses all styled puzzles toward "medium,"
  consistent with out-of-distribution confusion rather than genuine difficulty inflation.

**Verdict:** too data-starved (n=14 train, n=3 styled-hard) to answer reliably. No
reliable effect detected. Banked honestly. See STEP3_findings.md.

## Step 4 — Attention heads (CONFIRMED as correlational)

Found attention heads that attend disproportionately to style tokens over content
tokens. Required careful handling of attention sinks (sequence-start + first-content
anchors dominate raw attention). Method: average attention across all query positions,
sinks excluded, always eyeball-check the top head.

Validated via two controls: (1) plain comparison — real style heads refocus on content
when style is removed; (2) second-style confirmation — kept only heads style-specific
across BOTH archaic and poetic.

**Confirmed style heads: (21,1), (21,2), (23,2), (26,4)** — clustered in layers 21–26,
overlapping the Step-2 divergence band. NOTE: attention is CORRELATION, not causation
(tested in Step 5).

## Step 5 — Activation patching (CAUSAL)

**5b/5c — whole-state patch, single puzzle:** replacing the styled last-token
activation at layer 21 with the matched plain activation cut reasoning 416 → 123 (70%).

**Robustness (n=7 high-inflation puzzles, 3 styles):** mean reduction **67%**, range
33–90%, ALL positive. Effect is reliable specifically when there is substantial
inflation to remove; on low-inflation puzzles the intervention is weak or destabilizing
(consistent with the patch healing in proportion to how style-corrupted the state is).

**Layer sweep:** reduction is FLAT (~80%) across layers 15–27 — no single peak.
Early layers (1–10) are erratic/unstable. The stable causal band is **~layer 12–27**.

**5d — head-level patch (the decisive test):** patching ONLY the four Step-4 style
heads gave a mean ~14% reduction (48%, None, -20%) vs ~80% for whole-state on the same
puzzles. **The style heads alone do NOT cause the inflation.**

---

## Overall conclusion

1. **Phenomenon is real and robust:** styling inflates reasoning 3.6–5x across three
   styles, same logic, on an open model.
2. **The cause is DISTRIBUTED, not localized:** whole-state patching works flatly across
   layers 12–27; individual style heads do not reproduce the effect. The inflation
   signal is a distributed property of the mid/late residual stream.
3. **Attention heads that attend to style are witnesses, not culprits:** Step 4 found
   real style-attending heads, but Step 5d showed they are not the causal driver. This
   is the project's key nuance — it separates a tempting correlation from the actual
   causal picture.
4. **Convergence:** logit-lens divergence (Step 2), style-head clustering (Step 4), and
   the causal band (Step 5) all implicate mid-to-late layers — but the causal work
   shows the mechanism is spread across the residual stream there, not pinned to a
   circuit.

## Honest limitations

- Single model, single size (1.5B), easy base puzzles only.
- Small samples throughout (n=3–7 in causal tests); directions are clear, magnitudes noisy.
- Matched-plain transplant carries the plain puzzle's content (mild confound); a
  mean-plain or symmetric activation would be cleaner.
- Step 3 unresolved; would need a much larger, cleaner difficulty dataset.

## Files
- STEP3_findings.md, STEP4_findings.md — detailed step logs
- reasoning_tools.py — measure_reasoning() harness
- logit_lens.py, find_heads_fixb.py, plain_comparison.py — analysis
- patch_v2.py, robustness.py, layer_sweep_patch.py, head_patch.py — causal experiments