# The word-memory model: how it works, where it falls short, how to improve it

This document reviews the per-word memory model at the heart of `bilangsim`: how
agents remember, reinforce and forget the words of each language. It covers how
the model works today, its shortcomings, how they distort each agent's
language trajectory, and options for fixing them.

It complements:
- [`legacy_language_model_analysis.md`](legacy_language_model_analysis.md): the
  language model as a whole (Zipf sampling, MAXIMIN, code-switching).
- [`components/agent.md`](components/agent.md) and
  [`components/ic_generator.md`](components/ic_generator.md): code reference for
  the live learning rule and the initial-conditions (IC) recipe.

> **Status (Oct 2026):** analysis only. No code was changed. Under the project's
> phasing, the fixes proposed here belong to the model-correctness plan. All
> numbers come from the current code. Some are computed exactly from the
> formulas, and some are measured in an instrumented simulation (400 agents,
> 2 clusters). [Appendix A](#appendix-a--how-the-numbers-were-produced)
> explains how to reproduce them.

---

## 0. Summary

Each agent stores, for each language store (`L1`, `L2`, `L12`, `L21`) and each
of 500 compressed word slots, a **stability** `S` and the time `t` since the word
was last used. Together they give a **retrievability** `R`, the probability of
recalling the word. One number is derived from these arrays: `pct`, the share
of slots with `R > 0.9`. **Every social decision in the model reads `pct`**: the
mono/bilingual label, the conversation language (MAXIMIN), the language parents
pass to a newborn, family languages and friendships. Any distortion in the
memory model therefore reaches the macro results directly.

The main findings, grouped by theme:

| ID | Finding | Severity | Main consequence |
|---|---|---|---|
| **M1** | The SuperMemo stability formula is used as an *additive* increment, with its constants divided by 10. The published formula is a *multiplicative* factor. | High | `S` becomes a scaled lifetime word count. Spaced repeated use adds almost nothing after the first encounter. |
| **M2** | Repeated uses of a word within one step count as independent full-strength reviews, and the update is applied twice (≈ `2n − 1` uses). | High | 10 uses crammed into one 10-day step make a word **15× more durable** than 10 uses spread over a year, which reverses the spacing effect. Results depend on how the scheduler bundles conversations. |
| **M3** | Retrieval failures have no cost, and forgotten words get the *largest* gain when re-encountered. | Medium | Weak languages bounce back too easily. Memory has no lapses. |
| **M4** | The code-switch cascade reinforces the wrong trace. A word found in `L2` while speaking `L1` creates a fresh, strong `L12` trace that counts as **L1** knowledge. | Medium | `L1` `pct` is inflated for code-switchers, and their `L2` is under-reinforced. |
| **M5** | Forgetting is a single exponential. There is no long tail ("permastore"), and age at learning does not matter. | High | A bilingual adult who stops using a language is reclassified after about 5 years while still recalling 98 % of running words. |
| **M6** | Memory parameters do not depend on age. | Medium | The model has no sensitive period and no ageing effects. Age differences can come only from exposure differences. |
| **M7** | `pct` counts word *types* equally (rare = frequent), uses a hard `R > 0.9` cut-off, and folds `L12` into `L1`. | High | Classification is driven by rare-word forgetting, not by the ability to communicate. |
| **M8** | A compressed slot's meaning (which real words it stands for) changes with the agent's age, but its memory does not. | Medium–High | Vocabulary growth comes for free. Talking to toddlers reinforces an adult's rarest words 7× more often than talking to adults. |
| **M9** | Receptive and productive knowledge share one trace and one decay. | Medium | The model cannot represent receptive-only (heritage) bilinguals well. |
| **M10** | `L1` and `L2` memories do not interact: no competition, no cognate sharing. | Medium | Dominance shifts happen only through decay and exposure volume. |
| **M11** | Encoding of new words is inconsistent: gates differ, a single recognised cognate blocks all other new-word learning in an utterance, and noticing is capped. | Medium | Learning speed depends on arbitrary implementation details. |
| **M12** | Time-discretisation artefacts: an off-by-one in `t`, freshly touched words counting as known, one `R` per call, and dependence on call order. | Medium | The same 10 tokens in one step give `S` = 9.4, 24.9 or 142.5 depending on how they are split into calls. |
| **M13** | The IC recipe and the live dynamics follow different memory physics. | High (reliability) | The start of a run is a transient. Children born in the simulation lag IC children until age 2, then overtake them by age 3. |

**Recommendation in one paragraph.** Before changing the dynamics, (1) move the
memory rule into one module that both the IC recipe and the live agents use, and
build a small *diagnostic harness*: single-agent scenarios checked against
empirical stylised facts (spacing effect, forgetting-curve shape, permastore,
adult vs child attrition, comprehension coverage). Then (2) record a
frequency-weighted knowledge measure alongside `pct`. (3) Apply the cheap
"corrected legacy" fixes (Option A) behind a flag. (4) Replace the update rule
with a power-law model that has lapses. The recommended choice is an
FSRS-style rule, the modern descendant of the SuperMemo model the 2019 notebook
cited, with ACT-R base-level learning as the alternative. Finally (5) make the
structural changes: fixed-meaning slots, age effects, and cross-language
interaction. Section 4 details each option with worked numbers.

---

## 1. How the memory model works today

### 1.1 What each agent stores

For each agent, store (`L1`, `L2`, `L12`, `L21`) and slot `i` (0…499):

| Array | Meaning | Unit |
|---|---|---|
| `S[i]` | **Stability**: how many steps after its last use the word stays at least 90 % retrievable. | steps (1 step = 10 days) |
| `t[i]` | Steps since the word was last used. | steps |
| `R[i]` | **Retrievability** `= exp(−k·t/S)` with `k = ln(10/9) ≈ 0.105`. Read as the probability of recalling the word now. | 0…1 |
| `wc[i]` | Lifetime count of tokens of this word that the agent processed. | tokens |
| `mem_eff[i]` | Encoding threshold: how many tokens of an unknown word must be met before it enters memory. Drawn as `randint(3, 7)`, i.e. 3–6. | tokens |
| `pct[age]` | Share of the 500 slots with `R > 0.9`. For `L1` it uses `max(R_L1, R_L12)`; for `L2`, `max(R_L2, R_L21)`. | 0…1 |

**How to read `S`.** With the exponential curve, `R` is 0.9 when `t = S`,
0.81 at `t = 2S`, 0.59 at `t = 5S`, 0.35 at `t = 10S`, and 0.1 at `t ≈ 22S`. The
half-life is `≈ 6.6·S`. So a word with `S = 20` steps is recalled with ≥ 90 %
certainty for 200 days after its last use, about 59 % of the time 1000 days
after, and is practically gone (R < 0.1) after about 12 years.

### 1.2 The four operations

**(a) Decay, once per step** (`schedule.py:45-46`). For every word not used in
the previous step, `t += 1`. Then `R` and `pct` are recomputed.

**(b) Reinforcement on use** (`update_words_memory`, `agent.py:670-723`). When an
agent processes `n` tokens of a word in one call, with strength `ds`
(`ds = 1` speaking or studying, `ds = 0.1` listening):

```
g(S, R) = 7.6 · S^(−0.023) · e^(−3.1·R) − 0.2        # numba_comp_delta_S
S ← S + ds · n     · g(S, R)                          # first pass
S ← S + ds · (n−1) · g(S, R)                          # second pass, same old R
t ← 0
```

`g` depends strongly on `R` and weakly on `S`:

| `R` at time of use | `g` (S = 1) | `g` (S = 100) | `g` (S = 10 000) |
|---|---|---|---|
| 0.0 (forgotten) | 7.40 | 6.64 | 5.95 |
| 0.5 | 1.41 | 1.25 | 1.11 |
| 0.9 | 0.27 | 0.22 | 0.18 |
| 1.0 (just used) | 0.14 | 0.11 | 0.08 |

In plain words, using a word you had forgotten adds about 7 steps (70 days) of
stability per token. Using a word you know well adds about 0.1–0.25 steps per
token.

**(c) Encoding of unknown words** (`process_unknown_words`, `agent.py:563`).
When listening, a word with `R ≤ 0.1` counts as *unknown*. It can enter memory
in one of two ways:
- *Recognition*: it resembles a word the agent knows in the other language
  (random "edit distance" < 2).
- *Identification*: it is "noticed". At most `binomial(num_days, max(share
  understood, 0.1))` new tokens are noticed per call.

Its `wc` grows. Once `wc` reaches `mem_eff`, the word goes through (b).

**(d) Retrieval when speaking** (`pick_vocab`, `agent.py:933-961`). The agent
draws word intents from the age-specific Zipf CDF. Each *distinct* word is
recalled with probability `R`, using one random draw that decides for all its
tokens. Failures cascade `L1 → L12 → L2`. Words found only in `L2` are spoken,
and written into memory, as `L12`. The speaker then reinforces what it said with
`ds = 1`.

### 1.3 Where memory meets behaviour

`pct` is the only bridge between memory and social behaviour:

| Consumer | Use of `pct` |
|---|---|
| `update_lang_switch` (`agent.py:265`) | Language type 0/1/2: bilingual while both `pct ≥ 0.2`. |
| `get_conv_params` (`model.py:338`) | MAXIMIN: "real monolingual" if the other language's `pct < 0.025`; can speak it if `pct ≥ 0.1`. |
| `get_newborn_lang` (`model.py:247`) | A bilingual parent picks the language for the child with probabilities ∝ its two `pct`s. |
| `networks.py:163` | Relatives use the baby's language if their `pct > 0.1`. |
| `check_friend_conds` (`agent.py:1083`) | Friendship needs `|pct₁ − pct₂| ≤ 0.3` in the common language. |
| `pick_vocab` step 2 | Chooses the `L1` vs `L12` store by comparing `pct`s (broken by the `L12` `pct` bug). |
| `DataProcessor` | The reported knowledge per agent. |

### 1.4 Worked example: one word in a learner's life

Ahmed is 30, a monolingual `L1` speaker who moves to a cluster where `L2`
dominates. Follow one mid-frequency `L2` slot, say slot 120, which stands for a
word band such as "invoice".

1. **First weeks.** He hears the word but it is unknown (`R ≈ 0`). In some steps
   the identification route notices it. After 3–6 noticed tokens (his
   `mem_eff`), it enters memory.
2. **First memory update, heard once in that step.** `S = 0.01 + 0.1·8.25 ≈ 0.83`
   steps. At the next step `R = 1.00` (see M12), then 0.88, and after 100 days
   about 0.3. After about 6 months (`t > 22·S ≈ 18` steps) `R < 0.1`, so the word
   is "unknown" again.
3. **Same word, heard 10 times in one step** (a busy week of meetings):
   `S ≈ 14.6` steps. It stays ≥ 90 % retrievable for about 5 months, and `R` is
   still 0.77 a year later.
4. **Same 10 hearings spread out**, one per step (every 10 days): `S ≈ 0.96`, and
   the word is lost within months. One hearing per year for 10 years gives
   `S ≈ 4.1`.
5. **If he says it himself 3 times in one step:** `S ≈ 38.5`, so the word is
   ≥ 90 % retrievable for over a year from one 10-day period.

**The lesson of this example:** in the current model, how many times a word is
used *within a single 10-day step* matters far more than how many times it is
used *across months*. Memory research shows the opposite, and the following
sections explain why the model behaves this way.

---

## 2. Shortcomings

Each entry gives what the code does, why it is a problem, a concrete example,
and the impact on agents' language evolution.

### A. The reinforcement rule

#### M1. The SuperMemo formula is used additively, with its constants divided by 10

**What the code does.** The 2019 notebook (`lang_memory_model_research.ipynb`,
cell 194) cites <https://www.supermemo.com/articles/stability.htm> (Wozniak,
Gorzelańczyk & Murakowski, 2005, "Building memory stability through
rehearsal") as the source of the memory model. That article's equation (7.1) is

```
SInc = a · S^(−b) · e^(c·R) + d      R in percent, S in days
a = 76, b = 0.023, c = −0.031, d = −2
```

`SInc` is the stability **increase factor**: "quotient of stability S before
and after the repetition". It is applied as `S ← S · SInc` after a successful
repetition, and `S` is reset after a failed one. The article also notes that
"stability increase for retrievability of 100 % may be less than 1", which only
makes sense for a ratio.

The code uses `a = 7.6` and `d = −0.2`, exactly one tenth of the published
values, and **adds** the result to `S` measured in 10-day steps. The most likely
reading is that `SInc` was taken as "days of stability gained" and divided by
10 to convert days into steps.

**Why it is a problem.**
- **Spaced reviews barely grow memory.** In SuperMemo, a successful review at
  `R = 0.9` multiplies stability by about 2.2–2.7, which produces expanding
  review intervals: a few well-timed uses take a word from days to years. In the
  code, the same review adds about 0.25 steps (2.5 days) whatever `S` is. For a
  word with `S = 50` steps, a well-timed use adds 0.5 % instead of doubling it.
- **`S` turns into a scaled lifetime word count.** Frequent words are used at
  `R ≈ 1`, where `g ≈ 0.1`, so `S ≈ 0.07–0.26 × wc`. Example from the IC table,
  native speaker at 30 years:

  | Slot | Real words (at 30 y) | Lifetime tokens `wc` | `S` (steps) | `S` in years | `S / wc` |
  |---|---|---|---|---|---|
  | 0 | ranks 1–20 | 1 117 711 | 78 210 | 2 172 | 0.07 |
  | 10 | ranks 200–219 | 41 763 | 3 765 | 105 | 0.09 |
  | 100 | ranks 2000–2019 | 1 722 | 197 | 5.5 | 0.11 |
  | 499 | ranks 9979–9998 | 99 | 26 | 0.7 | 0.26 |

- **The first encounter dominates.** At `R ≈ 0`, `g ≈ 8`. Ten spaced uses of a new
  word end at `S ≈ 9.4` steps, 8.3 of which come from the first use.

**Impact.** Frequent words are robust and rare words are fragile, which is
qualitatively right. But the *timing* of use (regular contact vs. bursts vs. a
long gap followed by renewed use) has almost no effect beyond the first
encounter. Since language shift is largely about changes in the *pattern* of
use, this weakens the mechanism the model is meant to study. The numerical
mapping from exposure to retention also rests on an arbitrary rescaling (÷ 10)
and on the step length.

#### M2. Repetitions within a step count as independent full reviews, applied twice

**What the code does.** `n` tokens of a word in one call give
`S += ds·n·g(S,R)` and then `S += ds·(n−1)·g(S',R)` with the same old `R`.
Effectively that is `2n − 1` full-strength reviews, all at the `R` the word had
before the call. And `n` is large by design: `pick_vocab` draws
`words_per_conversation × num_days` tokens (`agent.py:892`). `num_days` (up to
10) means "this conversation happens on `num_days` of the 10 days in a step".
So 7 school days of a teacher's long talk (`'L'`, `num_days = 7`) become one
call with 700 tokens.

**Why it is a problem.** Uses massed in one sitting produce much less long-term
retention than the same number of uses spread out. This is the **spacing
effect**, one of the most robust findings in memory research (Cepeda et al.,
2006, a meta-analysis of 317 experiments). In the SuperMemo theory the model
cites, the second use of a word on the same day happens at `R ≈ 1`, where
`SInc ≈ 1`, so it adds almost nothing.

**Example.** A new word, 10 uses in total, `S` after the 10th use and `R` one
year later:

| Pattern | Speaking: `S` (steps) | `R` 1 y later | Listening: `S` | `R` 1 y later |
|---|---|---|---|---|
| All 10 in **one step** (current rule) | **142.5** | **0.97** | **14.6** | **0.77** |
| One per step (every 10 days) | 9.4 | 0.67 | 0.96 | 0.02 |
| One every 4 steps (40 days) | 9.7 | 0.68 | 1.3 | 0.06 |
| One per year | 13.7 | 0.76 | 4.1 | 0.39 |

The total exposure is the same in every row, yet cramming wins by a factor of
15, the opposite of the empirical ordering.

The *double application* on its own matters less than one might fear in the
steady state of a native speaker. Re-running the IC recipe with the live
`2n − 1` rule changes `pct` by less than 0.01 at every age and median `S` by
about 15 % (Appendix A, table A3). That is because frequent words are used at
`R ≈ 1`, where `g` is tiny. The harm comes in *learning episodes*: `R` is low,
`n` is large, and the multiplier turns one busy step into years of retention.

**Impact.**
- **Results depend on scheduling details.** The same weekly routine coded as
  "one call with `num_days = 7`" or "seven calls with `num_days = 1`" gives very
  different memories (see M12 for exact numbers).
- **Bursty contexts are over-weighted against sparse ones.** School and work
  (large `num_days`, long conversations) count for more than friends, shops and
  occasional contacts. Yet sparse, regular contacts are often where a minority
  language survives.
- **Learners get unrealistically durable traces from intensive episodes.** One
  busy step can produce an `L2` trace that lasts for years.

#### M3. No lapses: forgetting is never penalised and is often rewarded

**What the code does.** A failed retrieval (the random draw is above `R`) leaves
the store untouched. A listener who meets a word it has forgotten (`R ≤ 0.1`)
treats it as unknown, but its `wc` has long passed `mem_eff`. So as soon as it
is recognised or noticed again, the update runs at `R ≈ 0`, the maximum gain.

**Why it is a problem.** In the SuperMemo/FSRS family, a failed recall (a
*lapse*) *lowers* stability, and the post-lapse stability is far below what was
lost. Relearning is faster than first learning (Ebbinghaus's "savings"), but
forgetting is still a loss.

**Example.** A word with `S = 30` steps, heard 10 times in one step:
- While forgotten (`R ≈ 0`), it gains **+12.9 steps**.
- While still well known (`R = 0.95`), it gains **+0.32 steps**.

So the model rewards having forgotten the word 40×. For speaking, the expected
gain per attempt (`R · g`) peaks at `R ≈ 0.3–0.4` (0.73 steps) and falls to 0.16
at `R = 0.95`. Some advantage for effortful retrieval is realistic ("desirable
difficulties"), but here the failures that come with it cost nothing.

**Impact.** Weak or dormant languages recover too easily when exposure resumes.
This interacts with M13: agents start with weak `L2` traces and gain quickly in
the first months.

#### M4. The code-switch cascade reinforces the wrong trace and inflates L1

**What the code does** (`agent.py:937-961`). When a word fails in `L1` and `L12`
but is recalled from `L2`:
1. The `L2` trace, the one actually retrieved, is **not** reinforced.
2. The word is written into `L12`, where `R ≈ 0`. It therefore gets the largest
   possible first-encounter gain: with 3 tokens, `S ≈ 38.5` steps (≥ 90 %
   retrievable for over a year); with 10 tokens, `S ≈ 142.5`.
3. `L1`'s `pct` counts `max(R_L1, R_L12)` (`agent.py:751`), so the borrowed `L2`
   word now counts as **`L1` knowledge**.
4. `L12`'s own `pct` is never written (the documented bug), so step 2 of
   `pick_vocab` never chooses `L12` as the main store.

**Example.** A Spanish-dominant bilingual cannot recall a word in Spanish and
says it in Catalan 3 times during one step. From then on the word is "known in
Spanish" for more than a year, and the use does not strengthen their Catalan at
all.

**Impact.** `L1` `pct` is inflated for frequent code-switchers. `L2` retention
is under-estimated for bilinguals who use `L2` words inside `L1` talk.
Code-switching reinforces itself through a store that the knowledge measure
folds into the other language. Note that receiving the `L12` words is also
generous: 74 % of slots pass the `'mixed'` similarity test
(`binomial(10, 0.1) < 2`).

### B. Forgetting

#### M5. Single-exponential forgetting: no long tail, wrong age profile of attrition

**What the code does.** Between uses `S` is constant, so `R = exp(−k·t/S)` falls
by the same *proportion* every step. A word drops below 0.9 after `S` steps,
below 0.1 after `22·S`, and nothing slows the fall.

**Why it is a problem.**
- **Real forgetting curves have a long tail.** They are better described by
  power functions than by exponentials (Wixted & Ebbesen, 1991; Rubin & Wenzel,
  1996): rapid early loss, then a very slow tail. For `S = 50` steps, the
  exponential leaves `R` = 0.47 after 10 years and 0.05 after 40. A power law
  with the same `R(S) = 0.9` leaves 0.61 and 0.36.
- **Permastore.** Bahrick (1984) tested nearly 800 people who had learned
  Spanish at school up to 50 years earlier. Retention dropped for 3–6 years
  and then stayed essentially flat for 25+ years.
- **Age at disuse matters.** Adults who emigrate show only modest `L1` attrition
  even after decades (Schmid, 2011; de Bot, Gommans & Rossing, 1991). Children
  who lose contact before about 10–12 can lose their first language almost
  completely (Pallier et al., 2003, Korean adoptees in France; Bylund, 2009).

**Example.** Start from the IC state and stop all use of the language. Three
measures are shown: `pct` (share of slots with `R > 0.9`, which drives
classification), token coverage (share of *running words* whose slot has
`R > 0.9`), and recall per token (`Σ pᵢ·Rᵢ`):

| Agent | Measure | 0 y | +1 y | +5 y | +10 y | +20 y | +40 y |
|---|---|---|---|---|---|---|---|
| Native, stops at 30 y (`100_pct`) | `pct` | 1.00 | 0.65 | **0.21** | 0.12 | 0.07 | 0.04 |
| | recall/token | 1.00 | 1.00 | 0.98 | 0.97 | 0.95 | 0.93 |
| Native child, stops at 8 y (`100_pct`) | `pct` | 0.95 | **0.19** | 0.05 | 0.03 | 0.01 | 0.01 |
| | recall/token | 1.00 | 0.98 | 0.93 | 0.90 | 0.85 | 0.79 |
| Bilingual, 25 % exposure, stops at 30 y | `pct` | 0.82 | 0.30 | 0.08 | 0.05 | 0.03 | 0.02 |
| | recall/token | 1.00 | 0.99 | 0.96 | 0.94 | 0.90 | 0.86 |

In the model's own terms (`pct` vs `switch_threshold = 0.2`), the adult loses
the language in about 5 years and the child in under one year. Yet in terms of
recalling running words, the adult keeps 93 % after 40 years and the child
keeps 85 % after 20. Neither matches the evidence, which says adults keep most
of their L1 and children can lose it almost completely. Two things go wrong:
(a) forgetting has no tail, so `pct`, which is all about the rare-word tail,
collapses; and (b) nothing other than accumulated counts makes child-learned
memories more fragile.

**Impact.** Shift caused by disuse (migrants, moves between clusters, changed
jobs) shows up as rapid reclassification of adults, which likely
**over-estimates the speed of language shift**. Meanwhile children who switch
environment keep too much of the old language in token terms.

#### M6. No age effects in memory

**What the code does.** `g`, `ds`, `mem_eff` and the decay rule are identical
from age 0 to 100. The IC recipe has a post-65 decay (`S −= 0.01` per step,
i.e. 0.36 steps per year). That is negligible against median stabilities of
50–100 steps, and the live model does not apply it at all.

**Why it is a problem.** Age shapes lexical memory in several ways. Adults can
learn vocabulary faster at first (Snow & Hoefnagel-Höhle, 1978), but ultimate
attainment and resistance to attrition depend on age of acquisition (Johnson &
Newport, 1989; Bylund, 2009). Older adults have more retrieval failures, such
as tip-of-the-tongue states (Burke et al., 1991).

**Impact.** Intergenerational differences, for example children of immigrants
losing the home language while their parents keep it, can only emerge from
differences in exposure, never from memory.

### C. Measurement

#### M7. `pct` counts types, not tokens, with a hard 0.9 threshold

**What the code does** (`update_lang_pct_knowledge`, `agent.py:736-755`): `pct`
is the number of slots with `R > 0.9`, divided by 500 at every age. `L12` is
folded into `L1` and `L21` into `L2`.

**Why it is a problem.**
- **Comprehension depends on tokens, not types.** What matters is the share of
  *running words* one knows (coverage). About 95 % is needed for adequate
  comprehension and about 98 % for comfortable comprehension (Hu & Nation, 2000;
  Laufer & Ravenhorst-Kalovski, 2010). In the model's own adult distribution,
  slots differ enormously in weight:

  | Top slots known | 2.5 % (12) | 5 % (25) | **10 % (50)** | **20 % (100)** | 50 % (250) | 90 % (450) |
  |---|---|---|---|---|---|---|
  | Share of running words covered | 0.79 | 0.87 | **0.92** | **0.955** | 0.988 | 0.999 |

  So the "can speak" threshold (`pct ≥ 0.1`) corresponds to about 92 %
  coverage, and the bilingual threshold (`0.2`) to about 95.5 %. Meanwhile half
  of the `pct` scale (0.5 → 1.0) is about tail words that cover 1.2 % of speech.
- **A hard threshold turns a smooth probability into a cliff.** A word counts
  fully until `t = S` and then not at all, although at `R = 0.89` it is still
  recalled 89 % of the time.
- **Fixed denominator with shifting slot meaning (M8).** A `pct` of 0.5 means
  different vocabularies at 3 and at 30 years.

**Examples.**
- From M5: after 5 years without use, an adult native has `pct = 0.21` but token
  coverage 0.96 and recall per token 0.98.
- At 30 years, a bilingual with 25 % exposure has `pct = 0.82`, a monolingual
  1.00. The 18-point gap corresponds to token coverage 0.996 vs 1.000.

**Impact.** All the social decisions in §1.3 react to rare-word forgetting, to
the 10-day step and to the 0.9 constant. The thresholds 0.025, 0.1, 0.2 and 0.3
have no clear empirical meaning, so macro outcomes such as shift speed, the
share of bilinguals and intergenerational transmission inherit this
arbitrariness.

### D. Representation

#### M8. A slot's meaning changes with age; its memory does not

**What the code does.** The age-indexed CDF compresses the raw vocabulary
available at that age (`vocab_ceiling_curve`, 872 words at birth to 10 000 in
adulthood) into 500 equal-width rank bands (`Zipf_Mand_3S_CDF_comp`). Band
edges scale with the ceiling, so slot `i` covers raw ranks
`≈ [i·n/500, (i+1)·n/500)`, which depends on age:

| Age | Ceiling `n` | Slot 0 | Slot 10 | Slot 100 | Slot 499 | P(slot 499) |
|---|---|---|---|---|---|---|
| 0 y | 872 | ranks 1–2 | 18–19 | 175 | 870–871 | 1.7e-4 |
| 1 y | 1 045 | 1–3 | 21–22 | 209–210 | 1042–1044 | 1.9e-4 |
| 3 y | 1 632 | 1–4 | 33–35 | 327–329 | 1628–1631 | 1.3e-4 |
| 8 y | 5 250 | 1–11 | 105–115 | 1050–1060 | 5239–5249 | 4.8e-5 |
| 30 y | 9 999 | 1–20 | 200–219 | 2000–2019 | 9979–9998 | 2.6e-5 |

The memory arrays (`S`, `t`, `wc`) are indexed by slot and never remapped.

**Why it is a problem.**
- **Vocabulary growth is free.** A child who knows its 500 slots at 3 years,
  about 1 600 real words, "knows" the same 500 slots at 15, about 9 200 words,
  without meeting the extra 7 600. Only decay can take that knowledge away.
- **Stability earned while a slot was frequent persists after it becomes rare.**
  Slot 499 is 6.5× rarer at 30 than at birth.
- **Child-directed speech reinforces adults' rarest words.** When an adult
  speaks to a toddler, `pick_vocab` samples from the *toddler's* CDF
  (`min_age_interlocs`), but updates the adult's own slots. Under a 1-year-old's
  CDF, slot 499 is drawn 7× more often than under an adult's, so caring for
  toddlers reinforces an adult's rarest words.
- **The head of the distribution is coarse.** With linear bands, the 20 most
  frequent words (36 % of all tokens) share one memory trace. The top 12 slots
  carry 79 % of tokens. The part of the lexicon that matters most for
  comprehension has the least resolution.

**Impact.** Comparisons across ages (children vs adults, vocabulary growth,
age-of-arrival effects) are unreliable. M8 also feeds M7.

#### M9. Receptive and productive knowledge share one trace

**What the code does.** A single `R` serves both purposes. A word is understood
if `R > 0.1` and produced with probability `R`. That gives a word-level
asymmetry (a word at `R = 0.15` is always understood but produced 15 % of the
time), but both abilities decay together. Classification uses the same
`R > 0.9` measure with different thresholds (0.025 "understand", 0.1 "speak").
Listening strengthens memory at 0.1× the rate of speaking, and that is the only
other asymmetry.

**Why it is a problem.** Receptive vocabulary is larger and more resilient than
productive vocabulary. Heritage speakers often understand a language they can
barely speak (Polinsky & Kagan, 2007).

**Impact.** Receptive-only bilinguals, a key group for minority-language
sustainability, are poorly represented. Their status depends on the same
rare-word `pct` that governs speaking.

#### M10. No interaction between the languages in memory

**What the code does.** `L1` and `L2` arrays are updated independently. Slot `i`
is the *same concept* in both languages, since `pick_vocab` samples a concept
and then looks it up per store, which is a good basis for interaction. But the
only cross-language link is cognate recognition at encoding, and it relies on
random edit distances (M11).

**Why it is a problem.** Bilingual lexicons compete. Immersion in `L2` makes
`L1` words harder to retrieve (Linck, Kroll & Sunderman, 2009; the inhibitory
control model of Green, 1998). Cognates are learned faster and forgotten more
slowly (de Groot & Keijzer, 2000).

**Impact.** The model has no "my L1 is getting rusty" phase driven by
competition, and no protective effect of cognates. Dominance can shift only
through decay and differences in exposure volume.

### E. Encoding

#### M11. Inconsistent and order-dependent encoding of new words

**What the code does.**
- The encoding gate is `mem_eff ∈ {3,4,5,6}` per agent and language in the live
  model. The IC recipe uses a fixed `wc > 5`, i.e. 6.
- The comparisons are inconsistent: recognition uses `>=` (`agent.py:611`),
  identification uses `>` (`agent.py:642`), and studying uses `>=`
  (`agent.py:664`).
- **Recognition and identification are mutually exclusive** (`if/else` at
  `agent.py:607`). If *any* unknown word in an utterance is recognised as
  similar to a known other-language word, no other unknown word can be noticed
  in that call. The code's own TODO says so.
- Recognised words get `wc += 1` whatever their count. Identification notices at
  most `binomial(num_days, max(share understood, 0.1))` tokens per call. But once
  a word passes the gate, *all* its tokens in the call update `S` (back to M2).
- "Similar enough" is decided by random numbers: about 15 % of slots have
  `'original'` edit distance < 2 (`binomial(10, 0.3)`), and 74 % for the mixed
  stores.

**Example.** A bilingual child hears an `L2` conversation containing 30 unknown
words, one of which happens to fall in a "similar" slot and is known in `L1`.
Only that one word advances. The other 29 get no chance in this call. So the
more "cognate" slots a model draw happens to have, the *slower* non-cognates
can be learned.

**Impact.** The speed of learning a new language depends on arbitrary details
(the random cognate share, operator choice, call structure). Calibration across
scenarios is fragile.

### F. Time

#### M12. Discretisation artefacts: off-by-one, fresh words counted as known, one R per call, order dependence

**What the code does.**
- **Off-by-one.** A word used during step `n` gets `t = 0`. At the start of step
  `n+1`, `t += 1` skips it, because the step mask still marks it as used, and
  only afterwards is the mask reset (`schedule.py:45-47`). So it is still at
  `t = 0, R = 1` throughout step `n+1`. Every use buys one extra step of perfect
  recall. A word heard once (`S = 0.83`) shows `R = 1.00` at the next step
  instead of 0.88. The IC recipe does not have this offset.
- **Fresh words count as known.** Right after any update `t = 0`, so `R = 1`
  whatever `S` is, and the offset keeps it there for the next step. A word met
  for the first time (`S = 0.83`) therefore counts toward `pct` for two steps,
  and only then falls to `R = 0.88`. For learners, `pct` largely measures which
  words were *touched* in the last two steps, not which words are *known* (see
  the 16-year-old in M13).
- **One `R` per call.** All tokens in a call are evaluated at the pre-call `R`.
- **Order dependence.** The first call in a step that touches a word uses its
  real `R`, which gives a large gain if `R` is low. Later calls in the same step
  see `R = 1`.

**Example.** The same 10 speaking tokens of a new word inside one step:

| Split into calls | `S` after the step |
|---|---|
| 1 call × 10 tokens | **142.5** |
| 5 calls × 2 tokens | **24.9** |
| 10 calls × 1 token | **9.4** |

**Impact.** Memory depends on how the code happens to split behaviour into
calls, which is an implementation detail. The model is not invariant to the
step length either. That undermines confidence that a simulated effect, such as
"more school hours in L2", reflects the social mechanism and not the
bookkeeping.

### G. Consistency

#### M13. Initial conditions and live dynamics follow different memory physics

**What differs.**

| Aspect | IC recipe (`ic_generator.py`) | Live model (`agent.py`) |
|---|---|---|
| Update rule | `1 + (n−1)` uses | `n + (n−1)` uses (M2) |
| Listening vs speaking | All tokens at full strength (`F_S = 2` covers own speech, TV, radio) | Listening at 0.1× |
| Encoding gate | `wc > 5` | `wc ≥ mem_eff ∈ {3…6}` |
| `t` offset | none | off-by-one (M12) |
| Old-age decay | `S −= 0.01` per step after 65 | none |
| Exposure volume (adult) | 4 000 tokens/step at 100 % | ≈ 8 000 tokens/step enter memory updates (≈ 2 600 spoken, ≈ 5 450 heard); adolescents ≈ 13 000 |
| Infants | full-strength input from birth (≈ 200 tokens/step at 1.5 y) | nothing until 1 y (stages are gated on `age > 36`), then listening only (0.1×): ≈ 250 heard tokens/step averaged over all babies |

**Measured consequences** (instrumented 10-year run, 400 initial agents;
Appendix A):

- **Start-up transient.** Within 6 months, initial agents' `pct` jumps above the
  IC-table value for their age and bucket. The excess then fades slowly over
  years for strong buckets, but persists for the weakest:

  | Bucket | +0.5 y | +1 y | +2 y | +5 y | +10 y |
  |---|---|---|---|---|---|
  | `100_pct` | +0.06 | +0.06 | +0.04 | +0.02 | +0.00 |
  | `90_pct` | +0.03 | +0.03 | +0.02 | +0.01 | +0.00 |
  | `75_pct` | +0.06 | +0.06 | +0.05 | +0.02 | +0.00 |
  | `50_pct` | +0.10 | +0.10 | +0.10 | +0.06 | +0.03 |
  | `25_pct` | +0.11 | +0.12 | +0.11 | +0.09 | +0.07 |
  | `10_pct` | **+0.25** | **+0.26** | **+0.26** | **+0.23** | **+0.21** |

  The weak-language memories that the ICs encode are overwritten within months.
- **Language labels move very fast for learners.** 37 of 43 initial `L2`
  monolinguals (null `L1` memory) became bilingual during the run: a quarter
  within 4 months, half within 2.5 years. Most of the fastest were school-age.
  The fastest, a 16-year-old, went from `L1` `pct = 0` to 0.25 in its *first
  step* and was bilingual at step 3. That is a quarter of all `L1` word bands
  (≈ 2 500 real words) at ≥ 90 % recall after 10–30 days of exposure. M12
  explains most of it: any word that passes the encoding gate counts as known
  for two steps, whatever its stability.
- **Two cohorts with different rules.** Children born during the run hear
  nothing in their first year and only listen in their second. They then start
  speaking as `Child` at 2 years and gain massed, full-strength updates (M2).
  Measured `pct` of their stronger language, compared with IC children of the
  same age:

  | Age | 1 y | 1.5 y | 2 y | 3 y | 4 y | 5 y | 8 y |
  |---|---|---|---|---|---|---|---|
  | Born in the run (mean; n = 62 at 1 y, 19 at 8 y) | 0.00 | 0.04 | 0.16 | **0.71** | **0.91** | 0.95 | 0.98 |
  | IC table, `100_pct` | 0.24 | 0.33 | 0.38 | 0.56 | 0.73 | 0.86 | 0.95 |
  | IC table, `50_pct` | 0.13 | 0.18 | 0.25 | 0.35 | 0.49 | 0.61 | 0.83 |

  Children born in the run lag far behind until 2 years, then overtake even
  monolingual IC children by 3 years. Neither curve is validated against child
  vocabulary norms (V9 in §4.6), but the two cannot both be right.

**Impact.**
- The first months of every run are a transient; treat them as warm-up.
- More seriously, as the IC-born population is replaced over decades, aggregate
  knowledge measures drift for reasons unrelated to social dynamics. This
  confounds the language-shift signal the model is meant to measure.

---

## 3. Impact on the accuracy and reliability of language evolution

### 3.1 Individual trajectories: where the model is biased

| Scenario | What the model does | What evidence suggests | Bias |
|---|---|---|---|
| Native adult, continuous use | `pct ≈ 1`, stable | stable | ✓ realistic |
| Bilingual adult stops using a language (migration, job change) | `pct` falls below 0.2 in about 5 years, then reclassified (M5, M7) | modest attrition over decades | **attrition and shift over-estimated** |
| Child stops using a language before 10 | `pct` collapses within a year, but recall per token stays ≥ 0.85 for 20 years (M5, M6) | near-complete loss possible | token-level loss under-estimated; label flips too fast |
| Learner in intensive bursts (school, work) | very durable traces from a few steps (M2) | massed practice is weak | **burst learning over-estimated** |
| Learner with sparse but regular contact (friends, shops) | traces barely grow beyond the first use (M1, M2) | spaced practice is strongest | **regular contact under-estimated** |
| Frequent code-switcher | `L1` `pct` inflated by `L12`, `L2` not reinforced (M4) | use strengthens the language used | dominance mis-measured |
| Dormant language reactivated | largest gains at `R ≈ 0`, no lapse cost (M3) | savings, but lapses cost stability | recovery over-estimated |
| Child born during the simulation | far behind IC children until 2 y, ahead of them from 3 y (M13) | smooth growth | cohort artefact |
| Newcomer learning a language at school | a quarter of initial `L2` monolinguals become "bilingual" within 4 months (M12, M13) | years of exposure | **learning speed over-estimated** |

### 3.2 Population-level consequences

1. **Classification drives everything.** MAXIMIN, newborn language and
   friendships all read `pct` thresholds, and `pct` is dominated by the rare-word
   tail (M7). Macro outcomes such as shift speed and transmission rates
   therefore inherit the arbitrariness of the tail's decay rate, the 0.9
   threshold and the 10-day step.
2. **Exposure policies are mis-weighted.** Because of M1 and M2, the model
   rewards intensive, bundled exposure (school, work) relative to diffuse daily
   use. Scenario comparisons such as "immersion schooling vs. community use" are
   tilted before any social mechanism acts.
3. **Transient and cohort drift.** Without a warm-up the first months reflect IC
   overwrite, not dynamics. Over long runs, cohort replacement moves aggregate
   knowledge for reasons internal to the model (M13).

### 3.3 Reliability: invariances the model currently fails

A sound memory model should give the same answer for the same experience,
whatever the bookkeeping. Today it fails three checks:

| Invariance | Check | Result today |
|---|---|---|
| Chunking | same tokens, split into different numbers of calls | `S` = 9.4 / 24.9 / 142.5 (M12) |
| IC ↔ live | an IC agent vs. a simulated agent with the same exposure | different rules (M13) |
| Step length | 5-day vs 10-day steps | decay, off-by-one and `num_days` bundling all scale with the step (M2, M12) |

Until these hold, a difference between two scenarios may come from
implementation details and not from the social mechanism under study.

---

## 4. How to improve it

### 4.0 Ground rules for any change

1. **One memory module.** Move the update, decay and retrieval rule into
   `bilangsim/memory.py`, used by both `agent.py` and `ic_generator.py`. Select
   the rule with a parameter, keeping `'legacy'` as the default, so legacy
   behaviour stays reproducible. Regenerate ICs from the same rule and bump
   `GENERATOR_VERSION`.
2. **Diagnostic harness first.** Add single-agent scenarios, like the IC recipe,
   whose outputs are checked against the stylised facts in §4.6. Run every
   candidate rule through them before running the full model.
3. **Measure alongside `pct`, don't replace it yet** (§4.1). Keeping both makes
   the effect of each change visible.

### 4.1 Measure what matters (no change to the dynamics)

- **Frequency-weighted knowledge.** Record, per language,
  - token coverage `cov = Σᵢ pᵢ(age)·[Rᵢ > θ]`, and
  - expected recall per token `E = Σᵢ pᵢ(age)·Rᵢ`,

  where `pᵢ(age)` is the slot probability from the age CDF. These measures
  match the comprehension literature (95 % / 98 % coverage) and are naturally
  age-relative.
- **Caveat: the exposure calibration becomes visible.** In the current IC table,
  even the 10 % exposure bucket has coverage 0.987 at 30 years, i.e. near-native
  comprehension. Switching classification to coverage will therefore require
  recalibrating exposure and encoding at the same time. That generosity is
  already flagged in `components/ic_generator.md`.
- **Separate the code-switch stores.** Report `L12`/`L21` knowledge separately
  instead of folding it into `L1`/`L2` with `max`. Alternatively give it a
  weight below 1. Fix the `L12` `pct` bug.
- **Receptive vs productive.** Record a receptive measure (e.g. coverage at
  `R > 0.1`) and a productive one (coverage at `R > 0.9`, or expected recall).
  Use the receptive one for MAXIMIN's "understand" test.

### 4.2 Option A: "corrected legacy" (keep exponential and additive)

Minimal changes that fix the invariance failures without changing the model's
character:

| Fix | Change | Effect |
|---|---|---|
| A1 Single application | drop the second pass, or align live and recipe on `1 + (n−1)` | removes the ×2 |
| A2 Massed-aware within-step rule | first token at the real `R`, the remaining `n−1` at `R = 1`: `S ← S + ds·[g(S,R) + (n−1)·g(S',1)]`. If a call stands for `num_days` separate days, apply the first-token gain once per day at a 1-day gap. | the three chunkings of M12 all give `S = 9.4`; the 10 crammed uses of M2 give 9.4 instead of 142.5 |
| A3 Fix the `t` offset | at the start of each step, `t += 1` for **all** words (a word used in step `n` has `t = 1` in step `n+1`) | matches the recipe; removes the free step of `R = 1` |
| A4 Lapses | on failed retrieval in the intended store, `S ← max(φ·S, S_min)` (e.g. `φ = 0.5`); when a forgotten word (`R ≤ 0.1`) is relearned, start from a post-lapse `S` | forgetting costs something; relearning still benefits from `wc` (savings) |
| A5 Cascade targets | reinforce the store actually retrieved (`L2`); give the `L12` "borrowing" trace listening strength (`ds = 0.1`), not a first-encounter gain | M4 fixed |
| A6 One rule everywhere | the recipe calls the same module; regenerate ICs | M13 differences in rule, gate and offset removed |

**Pros:** small, local changes; calibration stays in the same ballpark;
chunking invariance is restored.
**Cons:** M1 (no benefit from spacing, `S` ≈ a word count) and M5 (no long
tail) remain. Treat Option A as a clean baseline, not the destination.

### 4.3 Option B: SuperMemo as published (multiplicative)

Implement the rule the 2019 notebook cites:

```
on successful use:   S ← S · max(SInc(S_days, R%), 1)
                     SInc = 76 · S_days^(−0.023) · e^(−0.031·R%) − 2
first encoding:      S ← S1 (e.g. 0.3 steps = 3 days)
on lapse:            S ← S_lapse (e.g. S1, or a fraction of S)
same-day repeats:    at most one review per word per day of activity
cap:                 S ≤ S_max (e.g. 200 years), to keep frequent words finite
```

**Worked numbers.** Ten spaced uses, with `S1` = 1 step:

| Spacing | Option B | Current rule |
|---|---|---|
| one per step | `S ≈ 16` | 9.4 |
| one every 4 steps | `S ≈ 42` | 9.7 |
| one per year | `S ≈ 215` | 13.7 |

Expanding intervals and the spacing effect come out naturally.

**Pros:** canonical spacing effect; faithful to the source the original author
chose; few parameters.
**Cons:**
- It was fitted on deliberate flashcard reviews of single items, not on
  incidental exposure to 20-word bands.
- Frequent words hit the cap immediately.
- Forgetting is still exponential (M5).
- It needs recalibration. Re-running the IC recipe under this rule, the 25 %
  exposure bucket reaches `pct = 0.98` at 15 years, against 0.76 today.

### 4.4 Option C: power-law forgetting with lapses (recommended direction)

#### C1. FSRS-style DSR model (recommended)

FSRS (Free Spaced Repetition Scheduler) is the modern, open-source descendant
of the SuperMemo two-component model. It tracks **D**ifficulty, **S**tability
and **R**etrievability per item, and its parameters are fitted on very large
review datasets (Ye, Su & Cao, 2022; open-spaced-repetition wiki). Its core
formulas (FSRS-4.5 to 6) are:

```
R(t, S) = (1 + f · t/S)^(−w)          f chosen so that R(S, S) = 0.9
                                       (FSRS-4.5/5: w = 0.5, f = 19/81; FSRS-6 fits w)
success:   S' = S · (1 + e^(w8) · (11 − D) · S^(−w9) · (e^(w10·(1−R)) − 1))
lapse:     S' = w11 · D^(−w12) · ((S+1)^(w13) − 1) · e^(w14·(1−R))
same day:  S' = S · e^(w17·(G − 3 + w18)) · S^(−w19)        (FSRS-6)
```

How it maps onto `bilangsim`:
- `S`, `t` and `R` already exist. Add one array `D` per store. `D` can be
  initialised from the slot's frequency band, cognate status and the distance
  between the languages.
- Successful production counts as a "Good" review and failed retrieval as an
  "Again" (lapse). Listening while understanding is a weaker review: scale the
  growth term by `ds`, as the model already does.
- The same-day formula handles within-step repetitions (M2, M12) directly.

**Why it fits.** The power law gives a long tail, so knowledge plateaus instead
of collapsing (M5). With `S = 50` steps, `R` is 0.61 after 10 years and 0.36
after 40, against 0.47 and 0.05 for the exponential. Lapses fix M3, and growth
depends on `R`, which gives the spacing effect and fixes M1.

**Costs.** One extra array. The review "grades" must be mapped onto
conversation events. Flashcard-fitted defaults must be checked against the
language stylised facts in §4.6, and possibly refitted.

#### C2. ACT-R base-level learning (alternative)

ACT-R's activation (Anderson & Schooler, 1991) is `B = ln(Σⱼ tⱼ^(−d))` over past
uses `j`, and the recall probability is `P = 1 / (1 + e^(−(B − τ)/s))`, with
`d ≈ 0.5`. Petrov's (2006) approximation needs only three numbers per word:

```
B ≈ ln( t₁^(−d) + (n − 1)·(L^(1−d) − t₁^(1−d)) / ((1 − d)·(L − t₁)) )
    n  = lifetime uses   (= wc, already stored)
    t₁ = time since last use (= t, already stored)
    L  = time since first use (one new array)
```

**Pros:**
- Frequency, recency and power-law forgetting all follow from one equation.
- Anderson & Schooler showed that word use in the environment follows the same
  power laws as memory. That makes it a natural fit for an exposure-driven ABM.
- Widely used for vocabulary learning (Pavlik & Anderson, 2005).

**Cons:**
- Plain base-level learning does not reproduce the full spacing effect. That
  needs Pavlik & Anderson's activation-dependent decay `dⱼ = c·e^(mⱼ) + α`,
  which in principle needs per-use history. A short history, such as the last
  2–3 uses plus an aggregate, is a workable compromise.
- It has no explicit lapse mechanism.

#### C3. Two-component trace (fallback)

Use a fast and a slow trace, e.g.
`R = w·e^(−k·t/S_fast) + (1 − w)·e^(−k·t/S_slow)`, where spaced uses consolidate
into the slow trace. This produces Bahrick-like plateaus with a small change
from today's code, but it is less standard and has more free parameters. Use it
only if C1 and C2 fail the harness.

**Recommendation.** Choose **C1 (FSRS-style)**. It keeps the conceptual model
the project started from (stability, retrievability and SuperMemo lineage) and
fits the existing arrays. It adds lapses, power-law forgetting and a principled
same-day rule. Use **C2** if grounding in environment statistics matters more
than continuity.

### 4.5 Structural improvements (independent of the update rule)

| Change | What | Fixes |
|---|---|---|
| **S1 Fixed-meaning slots** | Compress every age's distribution with the *adult* band edges, so slot `i` always means ranks `[20i, 20i+20)`, and give zero (or damped) probability to bands beyond the age ceiling. For example, a 1-year-old's CDF has mass only on slots 0–52, and later slots must be learned. Consider log-spaced bands so the high-frequency head gets more slots and the tail fewer. | M8; vocabulary growth becomes learning. **Requires** coverage-based or age-relative classification (§4.1) and regenerated ICs. |
| **S2 Age effects** | Make first-encounter stability, the encoding gate and lapse severity depend on age, and make durability depend on *age at learning*. For example, child-learned traces that are not reinforced after about 10–12 years decay faster, and retrieval failures increase after about 70. Calibrate against Pallier/Bylund (children) vs Schmid (adults). | M5 (age profile), M6 |
| **S3 Cross-language interaction** | (i) *Competition:* when speaking, choose between a concept's `L1` and `L2` traces with a Luce rule, `P(L1) = A₁ / (A₁ + λ·A₂)`, so recent `L2` dominance slows `L1` retrieval. (ii) *Cognates:* replace random edit distances with a curated cognate map, and let cognate slots share part of their stability gains. | M10, M11 |
| **S4 Receptive vs productive** | Use separate thresholds on one trace (receptive `R_rec = R^κ` with `κ < 1`), or a second, slower-decaying receptive strength. Classify "understands" and "speaks" separately. | M9 |
| **S5 Infant input** | Let babies listen from birth (remove the `age > 36` gate for home input), and make the recipe's exposure mix (spoken vs heard, `ds`) match live behaviour. | M13 (cohorts) |
| **S6 Code-switching as a link** | Represent `L12`/`L21` as a per-slot *borrowing habit* that biases retrieval toward the other language, while recall and reinforcement use the real `L2` (or `L1`) trace. | M4 |
| **S7 Encoding cleanup** | Use one gate (same operator, same distribution in recipe and live), make recognition and identification non-exclusive, and base the noticing cap on tokens rather than `num_days`. | M11 |

### 4.6 Validation targets (the diagnostic harness)

Each target becomes a single-agent scenario with an acceptance band:

| # | Stylised fact | Source | Harness scenario |
|---|---|---|---|
| V1 | Spaced > massed for the same number of uses; the optimal gap grows with the retention interval | Cepeda et al. 2006, 2008 | same token count, different spacings → compare `R` after 1–5 years |
| V2 | Forgetting has a power-law-like long tail | Wixted & Ebbesen 1991; Rubin & Wenzel 1996 | stop use → fit the curve shape |
| V3 | Well-learned vocabulary reaches a "permastore" plateau | Bahrick 1984 | learn for 3 years, stop → plateau after 3–6 years |
| V4 | Adult L1 attrition is modest after decades | Schmid 2011; de Bot et al. 1991 | adult native stops L1 → coverage stays high, label stays |
| V5 | Childhood L1 loss can be near-complete | Pallier et al. 2003; Bylund 2009 | child stops L1 at 6–9 → coverage collapses |
| V6 | Frequent words are learned earlier and kept longer | general | per-slot `S` vs frequency |
| V7 | Comprehension needs 95–98 % coverage | Hu & Nation 2000; Laufer & Ravenhorst-Kalovski 2010 | thresholds defined in coverage units |
| V8 | Receptive > productive; heritage profile | Polinsky & Kagan 2007 | low productive use, sustained listening |
| V9 | Plausible vocabulary growth with age | Fenson et al. 1994 (CDI); Brysbaert et al. 2016 | known raw-word count by age |
| V10 | Internal invariances: chunking, step length, IC ↔ live | — | §3.3 checks give equal results within tolerance |

### 4.7 Suggested order of work

1. **Memory module + harness + extra measures** (§4.0, §4.1, §4.6). No change to
   the dynamics. It documents how far the legacy rule is from V1–V10.
2. **Option A behind a flag** (§4.2). Regenerate ICs from the same rule, add a
   warm-up, and compare macro results with legacy.
3. **Option C1 (or C2)** (§4.4). Fit to the harness, regenerate ICs, compare.
4. **Structural changes** S1–S7 (§4.5), one at a time, re-running the harness and
   the macro comparison after each.

---

## Appendix A — how the numbers were produced

All computations use the code on this branch, with `PYTHONHASHSEED=0`.

**A1. Formula-level numbers** (§1.2 `g` table; M1–M4, M12 examples; Option A/B
numbers). These come from direct evaluation of the live rule, which for one
word in one call is

```python
k = np.log(10/9); a, b, c, d = 7.6, 0.023, -0.031, -0.2
g = lambda S, R: a * S**-b * np.exp(c*100*R) + d
def live(S, R, n, ds):                    # update_words_memory, one word, one call
    S1 = S + ds*n*g(S, R)
    return S1 + ds*(n-1)*g(S1, R)         # second pass reuses the old R
```

After a call `t = 0`, so a later call in the same step sees `R = 1`. In a later
step, `R = exp(−k·(gap − 1)/S)` because of the off-by-one (M12). SuperMemo
values use eq. 7.1 of the cited article (`a = 76, d = −2`, `R` in %).

**A2. IC-table numbers** (M1 slot table, M5 attrition table, M7 coverage table,
M8 slot table). The table is built with
`ic_generator.load_or_build_lang_ics(zipf_generator.age_indexed_cdfs(500, 3600))`.
- Attrition keeps `S` fixed and adds `Δ·36` steps to `t`.
- Slot probabilities are `pᵢ(age) = diff(cdf[age])`.
- Coverage is `Σ pᵢ·[Rᵢ > 0.9]`.
- Recall per token is `Σ pᵢ·Rᵢ`.

**A3. Re-running the recipe under other rules** (M2, Option B). This is
`build_bucket_trajectory` with the update line swapped. Each rule uses a fresh
`RandomState(42)`, so values differ slightly from the cached IC table, which
draws its buckets from one shared stream. For example, `pct` at 5 years for
100 % exposure is 0.82 here and 0.86 in the table.

| Rule | 100 %: `pct` at 5 / 15 / 30 y | 100 %: median `S` at 30 y | 25 %: `pct` at 5 / 15 / 30 y |
|---|---|---|---|
| recipe `1 + (n−1)` (IC table) | 0.82 / 0.98 / 0.99 | 56 | 0.39 / 0.76 / 0.81 |
| live `n + (n−1)` | 0.82 / 0.98 / 0.99 | 64 | 0.39 / 0.76 / 0.81 |
| massed-aware (A2) | 0.80 / 0.97 / 0.99 | 54 | 0.38 / 0.76 / 0.80 |
| SuperMemo multiplicative (B) | 0.92 / 1.00 / 1.00 | 110 | 0.41 / 0.98 / 1.00 |

**A4. Instrumented simulation** (M13, §3.1). `BiLangModel(400,
num_clusters=2)`, stepped with `schedule.step()`. The script wraps
`ListenerAgent.update_words_memory` to record the tokens entering each update
by mode (speak / listen / read), and records each agent's `pct` every step.
Initial buckets are identified by matching each agent's initial `S` row against
`lang_ICs[p]['S'][age]`. "Drift" is the live `pct` minus the IC table's `pct`
for the same bucket at the agent's current age. The run takes about 100 s per
simulated year. These are single-run measurements (one seed), so they show
magnitudes and directions, not precise estimates.

*Side finding:* the first run (seeds `rand_seed = 709`, `np_seed = 7345`)
crashed after about 2.5 simulated years in `Young.stage_3 → listen()` with
`self.pos is None` (`agent.py:458`). This runtime bug is unrelated to memory.
The 10-year run (a fresh random seed) guarded that call in the measurement
script only: it skipped the 49 `listen()` calls that would have crashed. Agents
whose `pos` is `None` are being scheduled, which deserves its own
investigation.

## Appendix B — references

- Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory. *Psychological Science*, 2(6).
- Bahrick, H. P. (1984). Semantic memory content in permastore: fifty years of memory for Spanish learned in school. *JEP: General*, 113(1).
- de Bot, K., Gommans, P., & Rossing, C. (1991). L1 loss in an L2 environment: Dutch immigrants in France. In Seliger & Vago (eds.), *First Language Attrition*.
- Brysbaert, M., Stevens, M., Mandera, P., & Keuleers, E. (2016). How many words do we know? *Frontiers in Psychology*, 7. (Already cited in the code's docstrings.)
- Burke, D. M., MacKay, D. G., Worthley, J. S., & Wade, E. (1991). On the tip of the tongue: what causes word finding failures in young and older adults? *J. Memory and Language*, 30.
- Bylund, E. (2009). Maturational constraints and first language attrition. *Language Learning*, 59(3).
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks. *Psychological Bulletin*, 132(3).
- Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: a temporal ridgeline of optimal retention. *Psychological Science*, 19(11).
- de Groot, A. M. B., & Keijzer, R. (2000). What is hard to learn is easy to forget. *Language Learning*, 50(1).
- Fenson, L., et al. (1994). Variability in early communicative development. *Monographs of the SRCD*, 59(5).
- Green, D. W. (1998). Mental control of the bilingual lexico-semantic system. *Bilingualism: Language and Cognition*, 1(2).
- Hu, M., & Nation, I. S. P. (2000). Unknown vocabulary density and reading comprehension. *Reading in a Foreign Language*, 13(1).
- Johnson, J. S., & Newport, E. L. (1989). Critical period effects in second language learning. *Cognitive Psychology*, 21(1).
- Laufer, B., & Ravenhorst-Kalovski, G. C. (2010). Lexical threshold revisited. *Reading in a Foreign Language*, 22(1).
- Linck, J. A., Kroll, J. F., & Sunderman, G. (2009). Losing access to the native language while immersed in a second language. *Psychological Science*, 20(12).
- Pallier, C., et al. (2003). Brain imaging of language plasticity in adopted adults: can a second language replace the first? *Cerebral Cortex*, 13(2).
- Pavlik, P. I., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: an activation-based model of the spacing effect. *Cognitive Science*, 29(4).
- Petrov, A. A. (2006). Computationally efficient approximation of the base-level learning equation in ACT-R. *Proc. ICCM*.
- Polinsky, M., & Kagan, O. (2007). Heritage languages: in the 'wild' and in the classroom. *Language and Linguistics Compass*, 1(5).
- Rubin, D. C., & Wenzel, A. E. (1996). One hundred years of forgetting. *Psychological Review*, 103(4).
- Schmid, M. S. (2011). *Language Attrition*. Cambridge University Press.
- Snow, C. E., & Hoefnagel-Höhle, M. (1978). The critical period for language acquisition: evidence from second language learning. *Child Development*, 49(4).
- Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science*, 2(6).
- Wozniak, P. A., Gorzelańczyk, E. J., & Murakowski, J. A. (2005). Building memory stability through rehearsal. SuperMemo article, the source cited by the 2019 notebook; a popularised rewrite of their 1995 paper "Two components of long-term memory", *Acta Neurobiologiae Experimentalis*, 55. <https://www.supermemo.com/articles/stability.htm>
- Ye, J., Su, J., & Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling. *KDD '22*. FSRS formulas: <https://github.com/open-spaced-repetition/awesome-fsrs/wiki/The-Algorithm>
