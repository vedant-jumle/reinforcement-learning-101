# Wordle GRPO Training: Failure Analysis and Path Forward

**Status:** Active investigation. Win rate is 0% after multiple training runs and four rounds of fixes.
**Scope:** `word_games/` — GRPO fine-tuning of Qwen2.5-0.5B-Instruct on single-guess Wordle decisions.
**Last updated:** June 2026

---

## 1. What Is Happening

The setup: a 0.5B instruct model is fine-tuned with GRPO (TRL `GRPOTrainer` + Unsloth LoRA) to play Wordle. Each training example is a single board state, and the model produces one guess in the format `<think>...</think><guess>word</guess>`. A reward function scores the completion: penalties for format failures, invalid words, and repeats; small step rewards for letter matches and candidate-set reduction; a large terminal bonus for winning.

What we observe after substantial training time across several runs:

- **Win rate: 0%.** Not low — zero. The model never solves a game, in training or eval.
- **Earlier runs: ~96% invalid guess rate**, hidden by the eval fallback that substitutes a random word when parsing fails. Aggregate metrics looked plausible while the model was effectively not playing.
- **After reward fixes** (harsher format penalties, repeat penalty at -10, constraint-consistency think bonus, terminal reward raised to 5–20): the model produces valid, mostly constraint-consistent guesses — and still never wins.
- The model has learned exactly what the gradient actually rewarded: *produce a legal, plausible-looking guess*. It has not learned to *win*, because (as Section 3 explains) the winning signal never physically entered the gradient.

Four rounds of fixes have already landed, and it's worth being clear about what they did and didn't address:

| Phase | Commit | Fixed | Did it move win rate? |
|---|---|---|---|
| 1. Stability | `2baa965` | Added KL penalty (beta=0.04), temperature=1.0, repeat penalty | No |
| 2. Hardware | `c84972d`, `8eb4441` | bf16/fp16 crashes on V100 (final: let Unsloth handle precision) | N/A — crashes |
| 3. Behavior | `0b15b78` | Word repeats, dead think blocks; constraint-violation checking | No |
| 4. Reward architecture | `d53ed8b` | Terminal signal now dominates step rewards; penalties recalibrated | No |

Each fix was correct and necessary. None addressed the root cause, because the root cause is not in the reward function.

---

## 2. Why the Current System Fails

### 2.1 How these LLMs are designed, and what that means for downstream RL

A pretrained LLM is a distribution over *token sequences*, shaped by next-token prediction on internet text. Three consequences matter here:

**The model's units are subwords, not characters.** Wordle is fundamentally a character-position constraint game, but the model's native vocabulary is subword tokens — "crane" may be one token, or ["cr", "ane"]. When the model "wants" to output crane, almost all of its probability mass sits on multi-character tokens. Its probability on the single-character token "c" is tiny. This means:

- Character-level reasoning ("position 3 must be N") is far from the model's pretraining distribution. It can be scaffolded in the prompt, but it is not free.
- Any scheme that samples character-by-character from the raw distribution gets a near-noise distribution, unless letter probabilities are properly *marginalized* over all tokens starting with that letter (see Section 5.2).

**Instruct-tuning gives format-following, not strategy.** Qwen2.5-0.5B-Instruct can roughly follow "respond with `<guess>word</guess>`" — at a high failure rate at this scale. It has no prior for *Wordle strategy* beyond weak letter-frequency knowledge. Downstream RL on a small instruct model therefore has to teach three things simultaneously:

1. Output format (tags, 5 letters, lowercase)
2. Constraint tracking (greens/yellows/greys → legal candidate set)
3. Strategy (which legal candidate maximizes information / wins)

These are stacked: strategy reward is unreachable until constraint tracking works, which is unreachable until format works. A 0.5B model with no SFT warm start spends its entire training budget climbing the bottom of this stack. The published reference points back this up: cold-start GRPO was already described as "a high-wire act" at 4B (Gemma-3); the successful run (Qwen 7B, 0/10 → 7/10) used SFT *then* GRPO. We are attempting cold start at 14x smaller than the model where cold start already struggled.

**Downstream RL fine-tuning should be thought of as *redistributing* probability mass, not creating capability.** RL with a KL anchor (beta=0.04 against the frozen reference) can sharpen behaviors the base model already produces with non-trivial probability. It cannot summon behaviors the base model essentially never samples. If the base policy wins a game with probability ~0, GRPO will not find the win — it will optimize whatever *is* reachable, which is exactly the legal-guess plateau we observe. The KL penalty, necessary for stability, actively anchors the policy to a reference model that cannot play Wordle.

### 2.2 The quirks of GRPO that cause the failure

GRPO has no critic. Its only learning signal is **within-group relative advantage**: sample K completions (here K=8) for the same prompt, compute each reward, subtract the group mean, normalize. Tokens of above-average completions get pushed up; below-average pushed down.

This design has a brutal corollary: **a reward that never appears within a sample group contributes exactly nothing, regardless of its magnitude.** The terminal win reward was raised from +1..6 to +5..20 in the Phase 4 redesign. Irrelevant — if zero of the 8 samples win, the +20 never enters any advantage calculation. The gradient sees only the variance that *does* exist in the group: better vs. worse legal guesses, think-bonus differences, the occasional format failure. So the model gets steadily better at the distribution of things it already does, and the win remains invisible.

Three more GRPO-specific failure amplifiers:

- **Degenerate groups → zero gradient.** If all 8 completions get similar reward (all mediocre legal guesses, or early in training, all format failures), advantages are ~0 and the step is wasted. Low policy entropy makes this worse — and anything that collapses entropy (including a hypothetical "reward high letter probabilities" term) directly kills the learning signal.
- **High outcome variance, no value function.** Win/loss is binary and rare; a critic could smooth credit over states, but GRPO's group baseline only compares completions of the *same* prompt. States from which winning is impossible (most of our dataset — see below) can never produce a win-vs-loss contrast within a group.
- **Sequence-level reward, token-level update.** The scalar reward smears uniformly across all completion tokens — think block and guess alike. A completion with a good guess but garbage reasoning, and one with great reasoning but a bad guess, are indistinguishable at the token level. At 0.5B this slows everything down further.

### 2.3 The dataset makes wins unreachable

`word_games/dataset.py` generates training states by playing 0–5 **uniformly random guesses** and recording the board. Two structural problems:

**Wins are combinatorially impossible from most rows.** After random guesses, the consistent-candidate set typically still contains hundreds to thousands of words. The probability that the model's single guess is the target is roughly 1/|candidates|. With 8 samples per prompt, P(at least one win in a group) is effectively zero for the overwhelming majority of rows. This is the quantitative form of the GRPO problem above: **the terminal reward is not sparse, it is absent.**

**Distribution mismatch.** Random-play board states are not the states a competent policy visits (random guesses produce feedback patterns that sensible play never reaches). At eval time the model faces its own trajectory states — a distribution it was never trained on. Train/eval mismatch compounds the capability problem.

### 2.4 Summary of the causal chain

```
0.5B cold start (no SFT)
        │  base policy: P(win) ≈ 0 from almost every state
        ▼
random-play dataset → states with 100s–1000s of candidates remaining
        │  P(win in any of 8 GRPO samples) ≈ 0
        ▼
GRPO group advantage: win reward never sampled → never in gradient
        │  only reachable variance: legal vs. illegal, better vs. worse step score
        ▼
model converges to "always produce legal, constraint-consistent guess"
        = exactly what we observe. 0% wins is the optimum of the
          gradient signal that actually exists.
```

The reward redesign (Phase 4) was correct but addressed the wrong layer: it fixed what the model would learn *if* wins were sampled. They never are.

---

## 3. How the Problem Should Be Thought Of

Reframe from "the reward function is wrong" to **"the win must become a samplable event."** Everything else is secondary. Three principles:

**1. Reachability before reward shaping.** The design question for every training example is: *can at least one of the K samples plausibly hit the terminal reward?* If P(win per sample) is p, then P(win appears in a group of K) ≈ 1 − (1−p)^K. For K=8, you need p ≳ 0.05–0.1 for wins to appear regularly. That means training states should mostly have **≤ ~10–20 consistent candidates remaining**. Reachability is a property of the dataset, not the reward.

**2. Separate the three learning problems and solve them with the cheapest adequate tool.**

| Problem | Wrong tool (current) | Right tool |
|---|---|---|
| Output format | RL penalties (-8/-4) burning reward budget | Constrained decoding — eliminate the problem mechanically |
| Constraint tracking | Hoping it emerges from scalar reward | SFT on solver traces + structured CoT template in the prompt |
| Strategy | GRPO from states where strategy can't pay off | GRPO from reachable states, curriculum outward |

RL is the most expensive and least sample-efficient of the three tools. Spend it only on the part that genuinely needs it (strategy), after the other two layers are handled by cheaper means.

**3. Think in terms of what the gradient can see.** Before any run, ask: across one batch, what reward variance exists *within groups*, and what behavior does climbing that variance produce? If the answer is "slightly better legal guesses," that is what you will get — for the entire run. This single question would have predicted the outcome of every run so far.

---

## 4. Next Directions

Ordered by expected impact per unit of engineering effort.

### 4.1 Backward curriculum on candidate count (highest priority, lowest cost)

Change `dataset.py` to generate/filter states by **candidates remaining**, not uniform random guess depth.

- **Stage A:** states with 1–5 consistent candidates (board nearly solved). P(win per sample) is high; wins flood the GRPO groups; the terminal reward finally enters the gradient. The model learns "read constraints → guess the forced word."
- **Stage B:** 5–30 candidates. The model must now choose informative guesses, not just forced ones.
- **Stage C:** 30+ candidates / empty boards. Opening strategy.

Promotion criterion: advance a stage when win rate on that stage exceeds a threshold (e.g., 60%). Implementation: generate boards using *solver-quality* guesses (greedy candidate-minimizing via existing `filter_candidates`) rather than random ones — this both controls candidate counts and fixes the distribution mismatch. No new infrastructure; one change to dataset generation.

### 4.2 SFT warm start on solver trajectories

Build a greedy entropy/candidate-minimizing solver from `utils.filter_candidates` (a near-optimal Wordle player in ~50 lines). Play out full games, render each decision as a `<think>` (constraint table filled from the board) + `<guess>` pair, and SFT the model on these traces before GRPO. This is the published recipe that worked (SFT → GRPO, 0/10 → 7/10 at 7B) and it directly addresses the cold-start problem, which is more severe at 0.5B than at any scale previously attempted. After SFT, the GRPO reference model is also a *competent* anchor instead of a hostile one.

### 4.3 Trie-constrained decoding for the guess span

Eliminate format and invalid-word failures mechanically instead of training against them:

- Build a trie over the 5757-word list.
- During generation, detect entry into the `<guess>` tag; from there, mask logits so only tokens continuing some valid word are allowed. The think block remains unconstrained.
- Letter probabilities must be **marginalized**: P(next letter = c) = Σ P(token t) over all valid-continuation tokens t starting with "c" — never renormalize over just the 26 single-character tokens, which yields a near-noise distribution (the model's mass is on subword tokens). This is what `outlines`/`guidance` implement; use one or replicate it.
- **Caution for training-time use:** if rollouts sample from the masked distribution, the trainer's logprob computation must match (same masking applied when scoring), or the importance ratios are biased. TRL does not support this out of the box. Safe deployment order: eval-only first (replaces the random fallback and gives honest metrics), training integration later if needed.

Side effect: the -8/-4 format penalties become dead code, and 100% of the reward budget goes to strategy.

### 4.4 What *not* to do

- **Don't add a reward term for "high probability on letter tokens"** (from `worldle_idea.txt`). It is redundant once decoding is constrained, it rewards a mechanism rather than an outcome, and its optimum is entropy collapse — all probability mass on one fixed sequence — which zeroes GRPO group variance and kills the only learning signal the algorithm has.
- **Don't keep raising the win reward.** +20 vs +200 makes no difference to a reward that is never sampled.
- **Don't add more penalty types.** The model is already at the "legal, consistent guess" plateau; more penalties shape behavior below the plateau, not above it.

### 4.5 Secondary levers (after 4.1–4.3)

- **More generations per prompt** (`--num-generations 16+`) once wins are reachable — more group variance, better baselines. Pointless before then.
- **Honest eval metrics:** track fallback/parse-failure rate as a first-class number; consider ending eval games on format failure instead of substituting a random word, so the win rate measures the model and not the fallback.
- **Structured CoT template** in the system prompt (explicit per-position constraint slots) — partially done in the current prompt; keep tightening, since at 0.5B the model should fill in a scaffold rather than invent a reasoning format.
- **Scale up if the floor holds:** if Stage A of the curriculum still shows no learning at 0.5B with SFT warm start, the honest conclusion is a capability floor — move to 1.5B before spending more compute.
- **Then the original research question:** once standard Wordle trains, the variant curriculum (hard mode → Quordle → Absurdle) and the transfer questions become live again. They are unreachable until the base game works.

### 4.6 Suggested execution order

```
1. Solver (greedy candidate-minimizing)            ~50 lines, reuses filter_candidates
2. Dataset v2: solver-generated boards,
   bucketed by candidates-remaining                 dataset.py change
3. SFT on solver traces                             small SFT script, hours of compute
4. GRPO Stage A (1–5 candidates)                    existing train_grpo.py, new dataset
   → checkpoint: do wins appear in training logs?   ← the single most informative signal
5. Trie-constrained decoding in eval.py             honest metrics, kill the fallback
6. Curriculum stages B, C
7. num_generations ↑, then variants/transfer work
```

The checkpoint at step 4 is the experiment that matters: if wins appear in GRPO sample groups and win rate on near-solved boards climbs, the diagnosis in this document is confirmed and the rest is engineering. If not, the bottleneck is model capability, and the next move is scale, not more reward design.
