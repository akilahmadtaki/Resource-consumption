# Reasoning Gym (RG) — Findings

Source: Stojanovski et al., "Reasoning Gym: Reasoning Environments for RL with
Verifiable Rewards" (NeurIPS 2025). https://github.com/open-thought/reasoning-gym

Why we tried it: RG generates unlimited puzzles with a *difficulty parameter*, and its
paper explicitly separates "Difficulty Parameters" from "Stylistic Parameters". We hoped
this would give a continuous difficulty knob to map exactly where the styling effect
dies out (a dose-response curve), with prompt length decoupled from difficulty.

Setup: task `family_relationships`, family_size = 4 / 6 / 8, 5 puzzles each (15 total),
DeepSeek-R1-Distill-Qwen-1.5B, greedy decoding. Plain (unstyled) baselines only.

## Example puzzles (one per difficulty level)

    family_size = 4  [27 words]
    Henry is married to Karen. They have a child called Sebastian.
    Sebastian is married to Eleanor.
    What relation is Henry to Karen? Answer with a single word.
    ANSWER: husband

    family_size = 6  [35 words]
    Henry is married to Karen. They have a child called Sebastian.
    Sebastian is married to Eleanor. They have children called Theodore and Kai.
    What relation is Theodore to Sebastian? Answer with a single word.
    ANSWER: son

    family_size = 8  [45 words]
    Henry is married to Karen. They have a child called Richard.
    Richard is married to Hannah. They have children called Sky and River.
    Sebastian is married to Eleanor. They have a child called Hannah.
    What is Karen to Sky? Respond only with the word that describes their relationship.
    ANSWER: grandmother

## Results

| family_size | prompt words | mean reasoning tokens | loops |
|-------------|--------------|----------------------|-------|
| 4 | 27-30 | 316 | 1/5 |
| 6 | 35-37 | 257 | 0/5 |
| 8 | 45-48 | 284 | 2/5 |

## Finding 1: prompt length up does NOT mean reasoning length up

Prompt length grew ~70% (27 -> 48 words) while reasoning length stayed flat
(316 -> 257 -> 284 tokens), with no monotonic trend.

This is independent evidence against the alternative explanation that has shadowed the
whole project -- "styled prompts are longer, so maybe length alone drives the extra
reasoning." It corroborates the noise floor measured on the length-matched records
dataset (plain rephrasing: median 0.85-0.94x).

Caveat: here prompt length and task complexity rose together (more people = more
sentences = more relations), so length was not varied in isolation. The strict claim is:
increasing family_size does not increase this model's reasoning effort.

## Finding 2: RG's "difficulty parameter" is NOT the model's experienced difficulty

Doubling family size (4 -> 8) produced no increase in reasoning effort -- the "easiest"
level actually had the highest mean. So RG's designed difficulty knob does not map onto
how hard the model finds a task, at least for this generator and this model size.

This echoes Step 3, where object-count labels turned out to be a poor difficulty proxy
(12 of 18 puzzles reshuffled when relabeled by actual reasoning length). Designed
difficulty and experienced difficulty are different quantities.

## Finding 3: baselines are in the verbose regime, not the terse regime

We hoped RG's trivially simple puzzles (one-word answers: "mother", "husband") would
return us to the terse regime where the styling effect lives (hand-written easy puzzles
= 66 tokens). They did not: ~260-320 tokens, comparable to BBH.

Also notable: 3/15 runs (20%) failed to terminate even on these one-word-answer puzzles.

Implication: the 66-token baseline on our hand-written puzzles now looks like the
exception, not the rule. This 1.5B model is verbose on almost everything else we've
tried (BBH, teammate's records dataset, RG).

## Decision

RG was not pursued further for styling experiments: the difficulty knob doesn't control
reasoning effort, prompt length remains confounded with difficulty within a generator
(4: 27-30 words, 6: 35-37, 8: 45-48 -- no overlap), baselines sit in the verbose regime
where our effect is known to vanish, and the loop rate is high.

Not tested: other RG generators (syllogism, propositional_logic) may behave differently.

## Reproduce

    # RG needs Python >= 3.10, so it lives in a separate venv (.venv-rg)
    python3.11 -m venv .venv-rg && source .venv-rg/bin/activate
    pip install git+https://github.com/open-thought/reasoning-gym.git
    python step1_phenomenon/rg_generate.py     # -> data/rg_puzzles.jsonl

    source .venv/bin/activate                  # back to the model venv
    python step1_phenomenon/rg_baseline.py
