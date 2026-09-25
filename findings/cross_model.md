# Cross-Model Findings — Does Style-Induced Reasoning Inflation Generalize?

**Question:** Our main results (Steps 1-5) were on DeepSeek-R1-Distill-Qwen-1.5B.
Does the phenomenon replicate on other reasoning models?

**Method:** Same puzzle, same 3 styles, same measure_reasoning harness. Model swapped
via BADTHINK_MODEL env var (no code changes). Inflation = styled think length / plain
think length, computed within each model's own tokenizer.

## Results

| Style   | Qwen-1.5B | Qwen-7B | Llama-8B | Qwen3-8B |
|---------|-----------|---------|----------|----------|
| Plain (baseline) | 66 | 88 | 836 | 1121 |
| Poetic  | 5.0x | 8.6x | loop | 1.3x |
| Song    | 3.6x | 9.4x | 0.8x | 0.9x |
| Archaic | 3.7x | 1.8x | 0.7x | 1.1x |

(R1-distill models: greedy. Qwen3-8B: seeded sampling — its card forbids greedy decoding.)

## Headline finding: the effect is BASELINE-DEPENDENT

Models split by plain baseline (default reasoning verbosity):
- TERSE models (Qwen-1.5B, 7B: 66-88 tok) -> strong inflation, 3.6-9.4x.
- VERBOSE models (Llama-8B, Qwen3-8B: 836-1121 tok) -> no inflation (0.7-1.3x) or loop.

Styling adds a roughly fixed reasoning overhead (preamble + over-formalization +
self-verification). On a terse model that's several times the content (large ratio).
On a model that already spends 1000+ tokens by default, the overhead is priced in --
no headroom, ratio ~1x.

The effect tracks the GAP between default and style-triggered verbosity -- not
architecture, not training lineage.

## Four independent corroborations

1. Terse Qwen (1.5B, 7B): large effect.
2. Verbose Llama-8B (different architecture): effect gone.
3. Verbose Qwen3-8B (different training lineage, not R1-distill): effect gone.
4. BBH hard puzzles (earlier): baselines already 400-1200 tok -> ~1.0x.

## Sub-findings

- SCALE amplifies lyrical styles: 1.5B -> 7B, poetic 5.0->8.6x, song 3.6->9.4x, but
  archaic FELL 3.7->1.8x. Register-switching styles (poetic/song) amplify with scale;
  delay-only (archaic) attenuates. Echoes the Step-2 register-switch finding. (n=1,
  suggestive.)
- Poetic induces NON-TERMINATION on Llama-8B (clean poem, loops forever) though it
  completes on both Qwen models. Loop-induction is model-dependent.

## Limitations

- n=1 puzzle per cell; ratios illustrative, but the baseline split is large/consistent.
- Qwen3-8B used seeded sampling (required), not greedy.
- Four models, all <=8B, mostly Qwen lineage. A terse-baseline model from a different
  lineage would sharpen the claim (open question).
- Cross-model raw token counts use different tokenizers; only within-model ratios compared.

## Reproduce

    export BADTHINK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    python step1_phenomenon/experiment1.py
    python step1_phenomenon/exp1_remaining_styles.py
    # for Qwen3-8B also: export BADTHINK_SAMPLE=1  (forbids greedy decoding)
