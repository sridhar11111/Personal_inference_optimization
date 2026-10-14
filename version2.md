# On-Policy Layer Selection for LLM Depth Pruning — Research Plan (v2)

> **What changed from v1**
> 1. Per-step teacher–student agreement on the pruned model's own text (D^OP) is **no longer a selection score**. ReCal and ShortOPD show it can be low while the output is broken (teacher-confirmed repetition). It is now a diagnostic only.
> 2. The trajectory metric T is now computed **only on the non-repetitive part** of each rollout. Full-length T has the same blind spot (see B.4).
> 3. Selection now uses **degeneration-aware on-policy scores**, and Phase 1 decides which one to use by testing it against a **ground-truth layer ranking**.
> 4. Related work and baselines are updated with ReCal, SLEB, and Shrestha et al. (2026).

---

## Part A — The Research, Explained

### A.1 The problem

Depth pruning removes whole Transformer layers from an LLM. The hard part is deciding **which** layers to remove.

Existing methods score layers on **fixed reference text** (teacher forcing):

- **ShortGPT (Block Influence):** how little a layer changes its hidden state.
- **PPL / KL selection (Shortened LLaMA, Minitron-style):** how much removing the layer changes predictions on fixed text.
- **SLEB:** the same idea applied iteratively, re-scoring after each removal.

### A.2 Why that may be the wrong question

At deployment, the pruned model generates text from its **own** earlier outputs. Small mistakes change the text it conditions on later, so it ends up in states it was never scored on (exposure bias). ShortOPD shows the result is severe: removing 4 of 36 layers from Qwen3-4B with Block Influence devastates greedy generation while multiple-choice scores look fine.

### A.3 Why "just measure on-policy agreement" is not enough

The obvious fix is to score each layer on the pruned model's own rollouts by checking whether the full model agrees with its next-token predictions. ReCal and ShortOPD show this fails. On a broken prefix, the full model often **agrees** with the broken continuation:

- **Looping:** after "The answer is 36. The answer is 36. The answer is", the full model also predicts "36".
- **Wrong reasoning:** after "5 − 3 = 2 apples, then he gives 1 away, so he has", the full model also predicts "1".

So low agreement-based damage does **not** mean the generation is healthy. A good selection score must look at **trajectory-level outcomes**, not just next-token agreement.

### A.4 The hypotheses

> **H1 (phenomenon):** Layer rankings from teacher-forced scores (BI, one-step KL, multi-step teacher-forced divergence) are poor predictors of the true generation damage caused by removing each layer.
>
> **H2 (method):** A degeneration-aware on-policy score predicts true generation damage well, and selecting layers with it produces better pruned models at equal depth.
>
> **H3 (robustness):** The advantage survives the same recovery training, or reduces the recovery compute needed.
>
> **H4 (Indic):** The gap between teacher-forced and on-policy rankings is larger for Indic languages than for English.

### A.5 Why Indic languages

Indic scripts typically use **more tokens per word**, so the same answer is a longer autoregressive trajectory, which gives drift more room to build. Indic generation also has a failure mode English lacks: **language switching** (drifting into English or another script). Both should make the teacher-forced vs. on-policy gap larger, which would explain why English-calibrated pruning transfers poorly to Indic generation.

### A.6 Novelty: what exists and what we add

| Work | What it does | How we differ |
|---|---|---|
| ShortGPT, Shortened LLaMA, SLEB | Layer selection on fixed reference text | We score on the pruned model's own rollouts |
| ShortOPD (2026) | On-policy **recovery** after BI selection; documents teacher-confirmed repetition | We change **selection**, not recovery. ShortOPD leaves selection as plain BI |
| ReCal (Oct 2026) | Recovery-aware **calibration** for **width** (channel) pruning, using forward KL on teacher-supported predictions | We do **depth** selection, use the pruned model's own rollouts, and include a training-free setting |
| Shrestha et al. (2026) | Shows layer pruning badly hurts generative reasoning; recovery with self-generated responses | We use self-generated text for **selection**, not recovery |
| Multilingual iterative pruning (IJCNLP-AACL 2025) | Effect of calibration language on multilingual pruning | We study exposure bias and Indic generation failure modes |

**Our claim:** layer importance should be judged by degeneration-aware damage on the pruned model's own generations, and this matters more for Indic languages.

> ⚠️ **Before writing anything, read:** ReCal in full, plus the "training-aware structure selection" works it cites (Tang et al., 2026; Ai et al., 2026), and the selection method in Shrestha et al. (2026). If any of them already selects layers using on-policy generation quality, reposition around the Indic findings. Also verify XMerge and TAP.

---

## Part B — Definitions

### B.1 Notation

- **P**: full model; **Q_l**: model with layer l removed (identity skip)
- **x**: prompt; **y**: a generated continuation
- **JS**: Jensen–Shannon divergence between next-token distributions
- **m(y)**: the **effective length** of rollout y, i.e. the position where a repetition loop is first detected (following ShortOPD's idea), or the full length if there is no loop

### B.2 Ground-truth damage (the target every score must predict)

For each single-layer-removed model Q_l, measure its real quality on a **held-out validation set** of generation tasks:

```math
G_l = \text{Quality}(P) - \text{Quality}(Q_l)
```

Quality is task accuracy on verifiable tasks (math, code), plus a judge or reference-based score for summarization and open generation, in English and each Indic language.

**Meaning:** how much worse generation actually gets when layer l is removed. This is expensive, so we compute it only in Phase 1 to validate the cheaper scores. It is **never** used for selection on the test set.

### B.3 Teacher-forced damage (baseline score)

```math
D^{TF}_l(H) = \frac{1}{H} \sum_{k=1}^{H} \mathbb{E}_{y_{<k} \sim P}\Big[ JS\big(P(\cdot \mid x, y_{<k}),\ Q_l(\cdot \mid x, y_{<k})\big) \Big]
```

**Meaning:** damage on states the full model would visit. This is the strongest baseline we must beat.

### B.4 Truncated trajectory damage (candidate score 1)

Full-length trajectory likelihood has the same blind spot as next-token agreement. Inside a loop, the full model assigns **high** probability to each repeated token, so a long loop actually **lowers** the average per-token surprise. To avoid this, score only the part of the rollout **before** the loop:

```math
T_l = \mathbb{E}_{y \sim Q_l}\Big[ -\frac{1}{m(y)} \sum_{k=1}^{m(y)} \log P(y_k \mid x, y_{<k}) \Big] - \mathbb{E}_{y \sim P}\Big[ -\frac{1}{m(y)} \sum_{k=1}^{m(y)} \log P(y_k \mid x, y_{<k}) \Big]
```

**Meaning:** how unnatural, to the full model, the pruned model's text looks before it degenerates. It catches drift and odd word choices, but on its own it cannot see the loop or a wrong final answer. Those are covered by the next two scores.

### B.5 Degeneration rate (candidate score 2)

```math
R_l = \Pr_{y \sim Q_l}\big[\, y \text{ loops, hits the length limit, or switches language} \,\big]
```

**Meaning:** how often removing layer l makes generation break outright. It directly measures the failure modes that agreement-based scores miss. Language switching is the Indic-specific part.

### B.6 Outcome damage (candidate score 3)

On a **calibration** set of verifiable prompts (math with known answers, code with tests), kept separate from the validation and test sets:

```math
C_l = \text{Acc}_{\text{cal}}(P) - \text{Acc}_{\text{cal}}(Q_l)
```

**Meaning:** the most direct signal of whether reasoning still works. It is binary and noisy, so it needs a few hundred prompts and is only available for verifiable tasks.

### B.7 Combined score (candidate score 4)

```math
S_l = \tilde{T}_l + \tilde{R}_l + \tilde{C}_l
```

where the tilde means each score is rank-normalized across layers (so no weights need tuning). Use this only if Phase 1 shows the individual scores capture different failures.

### B.8 Diagnostics only (never used for selection)

- **On-policy agreement:** D^OP, the same as D^TF but on prefixes from Q_l.
- **Agreement blind-spot rate:** the fraction of rollouts where D^OP is low **but** the rollout loops, hits the length limit, or is wrong. This directly measures the ReCal/ShortOPD failure in our own setting.
- **Exposure ratio:** A_l = D^OP / (D^TF + ε), used to describe layers, never to rank them.

### B.9 Layer categories

| Teacher-forced damage | On-policy damage (T, R, C) | Category | Meaning |
|---|---|---|---|
| Low | Low | **Safely redundant** | Remove it |
| Low | High | **Exposure-fragile** | Local metrics wrongly call it safe. These are the layers this paper is about |
| High | Low | Locally important, robust | Usually kept anyway |
| High | High | Strongly important | Everyone keeps it |

---

## Part C — Experimental Setup

### C.1 Models

| Stage | Models | Why |
|---|---|---|
| Phases 1–2 | Two ~1–3B instruct models from **different families** | Cheap; two families rule out one architecture's quirks |
| Phases 3–5 | One or two ~7–8B models | Headline results |

Prefer models with reasonable ability in your three Indic languages.

### C.2 Three separate data splits

| Split | Used for | Never used for |
|---|---|---|
| **Calibration** | Computing T, R, C, D^TF for selection | Final results |
| **Validation** | Ground truth G_l in Phase 1; choosing the score | Final results |
| **Test** | Final reported results | Anything else |

Each split covers English plus your three Indic languages, with math (verifiable), code (English only), summarization, and open generation. Confirm that each benchmark actually covers your three languages before committing.

### C.3 Fixed choices (identical across all methods)

- **Removal operator:** identity skip for all methods, with no reconstruction unless every method gets it.
- **Rollouts:** report both greedy decoding and sampling at temperature 1.0; select using the setting that matches evaluation.
- **Rollout length:** long enough for real answers (e.g. up to 1024 tokens for math). Short horizons hide degeneration.
- **Repetition detector:** one fixed rule (e.g. an n-gram repeated k times in a row) applied to all methods.
- **Language detector:** one fixed script/language-ID tool for language switching.
- **Seeds:** at least 3 for all rollout-based scores.

---

## Part D — The Plan, Phase by Phase

Each phase lists **what to do**, **what to observe**, and **what it means**.

---

### Phase 0 — Infrastructure (≈ 1 week)

**What to do**

1. Implement identity-skip layer removal.
2. Implement BI, one-step KL, D^TF, SLEB-style iterative re-scoring, and random selection.
3. Implement rollouts with the repetition detector, language detector, effective length m(y), and caching of the full model's log-probs.
4. Implement T, R, C, and the D^OP diagnostic.

**What to observe**

- The BI ranking matches published patterns.
- Removing no layers gives zero for every damage score.
- The repetition detector agrees with your manual reading of 50 random rollouts.
- The time cost of scoring one layer.

**What it means**

- If BI does not reproduce, the pipeline has a bug. Fix it first.
- A bad repetition detector corrupts T and R, so validate it by hand. It is the new single point of failure.
- The cost per layer tells you whether greedy selection at 7B is feasible.

---

### Phase 1 — Ground truth and the kill test (≈ 1.5 weeks)

> This phase decides whether the project continues.

**What to do**

1. For every layer l of each small model, build Q_l and measure the **ground-truth damage** G_l on the validation set, separately for English and each Indic language.
2. Compute every score on the calibration set: BI, one-step KL, D^TF, T, R, C, and S.
3. For each score, compute the **Spearman correlation with G_l** across layers.
4. Compute the **safest-k overlap**: of the k layers each score would remove first (k = 4, 8), how many are among the k truly safest by G_l?
5. Compute the **agreement blind-spot rate** (B.8).

**What to observe**

- Which scores predict ground truth best, and by how much.
- Whether teacher-forced scores (BI, KL, D^TF) do worse than on-policy scores (T, R, C, S).
- Whether the gap is larger for Indic languages.
- How often D^OP is low while the rollout is broken.

**What it means**

| Observation | Interpretation | Action |
|---|---|---|
| D^TF predicts G_l about as well as T/R/C (e.g. Spearman within 0.05) | Teacher-forced scoring is enough. No exposure gap for selection | Go to Part E (check subgroups first) |
| On-policy scores clearly beat D^TF on generation tasks | **H1 confirmed.** The phenomenon is real | Continue |
| R or C beats T | Outcome failures (loops, wrong answers) matter more than drift | Use R/C, or S, in Phase 3 |
| T, R and C each catch different layers | Damage has several forms | Use the combined score S |
| Blind-spot rate is substantial | Confirms ReCal/ShortOPD in your setting and justifies dropping D^OP | Report it as a finding |
| Gap larger for Indic | **H4 supported** | Make this a headline result |

> **Why this design is strong:** instead of arguing which score *should* be better, we measure which one actually predicts real damage. Reviewers find this hard to dispute.

---

### Phase 2 — Characterise the phenomenon (≈ 2 weeks)

**What to do**

1. **Length sweep:** recompute scores with maximum rollout lengths of 128, 256, 512, and 1024 tokens.
2. **Task split:** short QA, math, code, summarization, and open generation.
3. **Stability:** 3 seeds × 2 decoding settings.
4. **Failure analysis of exposure-fragile layers:** for each such layer, record the first point where the pruned model diverges from the full model, and classify the failure (loop, truncation, wrong reasoning, language switch).
5. **Token-length link:** tokens per word for each language against the size of the gap.

**What to observe**

- The rollout length at which score rankings stop changing.
- Which tasks show the largest gap.
- Whether rankings are stable across seeds.
- Which failure types dominate, and whether language switching is common in Indic.
- Whether the gap grows with tokens per word.

**What it means**

| Observation | Interpretation |
|---|---|
| Rankings stabilize at some length | That is the cheapest rollout length worth using |
| Gap large on long-form and reasoning, small on short QA | Matches the theory, and explains why multiple-choice benchmarks hide the damage |
| Rankings unstable across seeds | Increase calibration size before Phase 3, or the method will not reproduce |
| Language switching dominates Indic failures | A concrete Indic-specific finding, and strong qualitative material |
| Gap grows with tokens per word | Direct evidence for the explanation in A.5 |

---

### Phase 3 — The selection method (≈ 2 weeks)

**What to do**

1. **Greedy degeneration-aware selection**, using the score chosen in Phase 1:

```text
Start with the full model
        ↓
Score every remaining layer with the chosen on-policy score
        ↓
Remove the safest layer
        ↓
Re-score remaining layers using rollouts from the *current* pruned model
        ↓
Repeat until the target depth is reached
```

2. **Cheaper variants** (needed at 7B+): score once, then re-score only the top-k candidates; use the shortest stable rollout length from Phase 2.
3. **Baselines** at equal depth (e.g. 32 → 28 and 32 → 24), all with identity skip:
   - Random selection (lower bound)
   - ShortGPT BI
   - One-step KL / PPL (Shortened LLaMA style)
   - SLEB (iterative re-scoring)
   - **D^TF with the same greedy re-scoring** (the most important baseline, because it isolates "on-policy vs. teacher-forced")
   - **Greedy D^OP** (shows the agreement blind spot hurts selection, not just measurement)
   - **ReCal-style probe adapted to layers** if feasible (forward KL on teacher-supported predictions)

**What to observe**

- Which layers each method removes.
- Test-set quality at each depth.
- Selection cost in GPU-hours.

**What it means**

| Observation | Interpretation |
|---|---|
| Our score beats greedy D^TF on generation, ties on multiple-choice | The predicted pattern: gains appear where exposure bias acts |
| Our score beats greedy D^OP | Confirms the agreement blind spot matters for selection, a direct answer to ReCal's concern |
| Our score beats BI but not D^TF | The gain comes from KL/greedy, not from on-policy scoring. Go to Part E |
| Cheap variant ≈ full greedy | The method is practical at scale |
| Gains grow with pruning ratio | Errors from multiple removed layers interact, which iterative re-scoring captures |

---

### Phase 4 — The recovery test (≈ 2 weeks)

**What to do**

A 2×2 experiment at one target depth:

| | No recovery | Same recovery (fixed budget) |
|---|---|---|
| **BI selection** | ✓ | ✓ |
| **Our selection** | ✓ | ✓ |

Use one standard recovery recipe for both, ideally SFT followed by on-policy distillation, which matches ShortOPD and ReCal. Evaluate at regular checkpoints to get recovery curves, and track the length-limit rate after recovery (ReCal's Figure 1c measure).

**What to observe**

- The final quality gap after recovery.
- The steps needed to reach a fixed quality level.
- The length-limit and repetition rates after recovery.

**What it means**

| Observation | Interpretation | Framing |
|---|---|---|
| Our selection still wins after recovery | Selection matters even with training | Strongest claim |
| Gap closes, but ours recovers faster | Selection saves recovery compute | "Same quality, X% less recovery" |
| Ours has fewer loops or length-limit hits after recovery | Better selection leaves less damage for OPD to fight, directly complementing ReCal | Strong link to the recent literature |
| No difference after recovery | Selection matters only without training | Training-free framing (Part E) |

---

### Phase 5 — Full evaluation (≈ 2 weeks)

**What to do**

Evaluate all pruned models on the **test** split:

- Static: MMLU, ARC, and an Indic QA/classification set
- Perplexity per language
- Reasoning: GSM8K and a multilingual math set
- Code: HumanEval/MBPP
- Long-form: summarization in English and Indic languages
- **Generation health:** repetition rate, length-limit rate, language-consistency rate
- Speed: tokens/sec at each depth (should be equal across methods at equal depth)

**What it means**

- Gains concentrated on generative tasks support the claim that **static benchmarks hide pruning damage**.
- Larger Indic gains and lower language switching support H4.

---

### Phase 6 (optional) — A cheap proxy (≈ 1–2 weeks)

Find a signal that predicts the chosen on-policy score **without rollouts** (for example, the full model's end-of-sequence probability, late-position teacher-forced divergence, or hidden-state drift). Measure its Spearman correlation with G_l. A good proxy makes the method cheap, which is valuable if Phase 3 gains are real but modest.

---

## Part E — If Results Don't Go Our Way

| Failure | First check | Fallback paper |
|---|---|---|
| **Phase 1: teacher-forced scores predict ground truth as well as on-policy scores** | Does the gap appear for long generation, high pruning ratios, or Indic only? | (a) A narrower paper on the subgroup where it holds; (b) a careful negative result: "teacher-forced selection is sufficient" (workshop / Findings); (c) the **original Indic pruning paper**, built on the per-layer ground-truth data already collected |
| **Phase 3: beats BI but not greedy D^TF** | Does it win on Indic or long-form only? | "Multi-step divergence beats cosine for Indic pruning", modest but publishable |
| **Phase 4: recovery erases the gain** | Is recovery faster, or are there fewer loops after recovery? | Training-free framing, a compute-savings paper, or the finding "selection matters little after on-policy recovery" |
| **Gains real but small** | Can a cheap proxy match them? | An efficiency paper built on Phase 6 |
| **Tang/Ai/Shrestha already do on-policy layer selection** | What do they leave out? | Reposition around the Indic findings and the ground-truth predictor comparison |

**Built-in protection:** the Phase 1 ground-truth ranking across two model families, several tasks and four languages is a useful benchmark in itself. Even a negative result is a characterised finding.

---

## Part F — Timeline

| Weeks | Phase | Decision point |
|---|---|---|
| 0 | Read ReCal, Tang et al., Ai et al., Shrestha et al. | Is the idea still open? |
| 1 | Phase 0: infrastructure | BI reproduces? Repetition detector validated? |
| 2–3 | **Phase 1: ground truth and kill test** | **Continue / pivot; which score?** |
| 4–5 | Phase 2: characterisation | Rollout length and task scope |
| 6–7 | Phase 3: selection method | Beats greedy D^TF and greedy D^OP? |
| 8–9 | Phase 4: recovery 2×2 | Which framing? |
| 10–11 | Phase 5: full evaluation | — |
| 12–13 | Phase 6 (optional) and writing | — |

---

## Part G — Paper Outline

1. **Introduction:** layer selection is judged on reference text, but deployment is autoregressive; even on-policy agreement is misleading.
2. **Related work:** layer selection (ShortGPT, Shortened LLaMA, SLEB), generative failure of pruned models (Shrestha et al., ShortOPD), recovery-aware pruning (ReCal and the works it cites), multilingual/Indic compression.
3. **Which scores predict real damage?** Ground-truth comparison across scores, models and languages; the agreement blind spot.
4. **Characterisation:** effects of rollout length, task and language; failure types; language switching; the token-length link.
5. **Method:** greedy degeneration-aware selection and its cheap variant.
6. **Experiments:** equal-depth comparison, the recovery 2×2, the full evaluation.
7. **Analysis:** why static benchmarks hide the damage; English vs. Indic.
8. **Limitations:** rollout cost, dependence on the repetition and language detectors, and reliance on verifiable tasks for C.

---

## Part H — Rules That Keep the Paper Defensible

1. Use the **same removal operator** for all methods.
2. Keep **calibration, validation and test splits separate**.
3. Make **greedy D^TF** and **greedy D^OP** the key baselines, not BI.
4. **Validate the repetition and language detectors by hand.**
5. Never select layers by agreement on the pruned model's own text alone, and say why (cite ReCal and ShortOPD).
6. Report **confidence intervals** and **selection cost**.
7. Describe results as **sensitivity to self-generated states**, not as causal error chains.
8. **Verify every citation** yourself.
