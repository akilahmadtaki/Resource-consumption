# Style-Induced Reasoning Inflation — Mechanistic Interpretability Project

Investigating how stylistic framing (poetry, song lyrics, archaic prose) makes a
reasoning model produce longer chain-of-thought — and localizing *where* and *what*
causes it inside the model.

**Model:** DeepSeek-R1-Distill-Qwen-1.5B
**Full write-up:** see `findings/all_together.md`

---

## TL;DR of results

- Styling inflates reasoning **3.6-5x** on the same puzzle (logic unchanged).
- The inflation signal is **distributed** across the mid/late residual stream (~layers 12-27).
- Activation patching the last-token state **causally removes ~67%** of the inflation
  (n=7 high-inflation puzzles, all positive).
- Style-attending attention heads exist but are **witnesses, not culprits** — patching
  them alone does NOT reproduce the effect.

---

## Setup

Requires a GPU (~8GB+ VRAM is plenty for the 1.5B model).

    python -m venv .venv
    source .venv/bin/activate
    pip install torch transformers accelerate scikit-learn numpy

    # redirect HuggingFace cache to large storage (recommended on shared servers):
    export HF_HOME=/path/to/large/storage/hf_cache

## How to run

**Always run scripts from the project root**, e.g.:

    python step1_phenomenon/experiment1.py

Not by cd-ing into the step folder — shared modules (in common/) and generated data
files (at the root) are located relative to the root. The helper file sitecustomize.py
automatically makes common/ importable, so no path setup is needed.

### Generate the data first (Step 3 onward)

The .pt / .npy data files are NOT committed to git (they're regenerable). Before
running Step 3+ analysis scripts, generate them:

    python step3_difficulty_probe/extract_activations.py   # -> plain_activations.pt
    python step3_difficulty_probe/measure_lengths.py        # -> plain_lengths.pt
    python step3_difficulty_probe/extract_styled.py         # -> styled_activations.pt
    python step3_difficulty_probe/make_labels.py            # -> length_labels.pt

## Repository layout

    common/                  Shared modules imported everywhere
    step1_phenomenon/        Reproduce the overthinking effect
    step2_logit_lens/        Layer-by-layer localization (logit lens)
    step3_difficulty_probe/  Linear probe for difficulty (INCONCLUSIVE)
    step4_attention/         Attention-head analysis
    step5_patching/          Activation patching (the CAUSAL tests)
    findings/                Write-ups (READ findings/all_together.md)

Main scripts per step:
  step1: experiment1.py       (plain vs styled, measures <think> length)
  step2: logit_lens.py        (decode residual stream per layer)
  step4: find_heads_fixb.py   (find style-attending heads)
  step5: patch_v2.py          (single-puzzle causal patch, 416->123)
         patch_high_inflation.py  (robustness, 67% mean reduction)
         head_patch.py        (heads alone are weak — not the cause)

## Suggested reading order for reviewers

1. findings/all_together.md — the whole story
2. Run step1_phenomenon/experiment1.py to see the effect
3. step5_patching/patch_v2.py — the core causal result

## Key caveats

Single model, single size, easy puzzles, small samples (n=3-7 in causal tests) —
directions clear, magnitudes noisy. See limitations in the findings doc.
