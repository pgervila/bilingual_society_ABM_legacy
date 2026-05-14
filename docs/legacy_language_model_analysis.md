# Language Learning Model in the Legacy ABM — Deep Analysis

This document examines how language acquisition, retention, code-switching, and bilingualism are modelled in `legacy_project/bilingual_society_ABM/bilangsim/`. It is the companion piece to `docs/legacy_vs_new.md`, which covers the codebase architecture overall.

## 1. Word Representation

**Compressed vocabulary.** Each agent's vocabulary is a fixed array of either 500 (spoken-only) or 1000 (spoken+written) "slots" — `vocab_red` in `model.py:86-88`. One slot stands in for many real words (~40:1 compression); slots are ordered by global frequency rank, so index 0 is the most common word and index `vocab_red-1` the rarest.

**Word frequency: Zipf-Mandelbrot.** Sampling lives in `zipf_generator/Zipf.py`. Three variants are implemented; the most realistic is a piecewise three-stage Zipf-Mandelbrot (lines 20-32) with different exponents per frequency band:

- α₁ = 1.16 for ranks 1–100 (the very common words)
- α₂ = 1.48 for ranks 100–2000
- α₃ = 1.866 for ranks > 2000

This captures the empirical fact that real word-frequency curves are not single power laws — head and tail follow different slopes. Sampling uses inverse-CDF with binary search (`randZipf`, lines 93-97): O(log n) per word.

**Age-specific vocabularies.** CDFs and initial conditions are precomputed externally and loaded from HDF5 (`model.py:114-115`), keyed by age in "steps". A 2-year-old's CDF reaches only the most common ~few hundred slots; an adult's covers the full vocabulary. This is how vocabulary *growth* is encoded — not by adding new entries, but by raising the ceiling on which Zipf ranks can be sampled.

## 2. Per-Agent Memory State

Each agent holds a `lang_stats` dict with **four keys**: `L1`, `L2`, `L12`, `L21`. The pure stores represent native L1/L2 knowledge; the mixed stores `L12` (L1→L2 switching) and `L21` (L2→L1 switching) hold code-switched material.

For each language, per-word NumPy arrays of length `vocab_red`:

| Array | Type | Meaning |
|---|---|---|
| `S` | float64 | Memory **stability** — how well-consolidated the word is |
| `t` | float64 | **Elapsed steps** since last activation |
| `R` | float64 ∈ [0,1] | **Retrievability** — instantaneous access probability |
| `wc` | int64 | **Word count** — encounters with this word |
| `pct` | float64 array [3600] | Aggregate language proficiency at each age step |
| `excl_c` | float64 | Linguistic exclusion counter |
| `mem_eff` | int32 | Repetitions needed before encoding (random ∈ [3,7]) |

### The Ebbinghaus / SM-2 memory model

The core retention equation (`agent.py:128-131`) is:

$$R = \exp\left(-k \cdot \frac{t}{S}\right) \quad \text{with } k = \ln(10/9) \approx 0.1054$$

This is the **Ebbinghaus forgetting curve**, parameterised by an SM-2-style "stability". The constant *k* is chosen so that when `t == S`, retrievability drops to exactly 0.9 — i.e. stability *S* is the time-horizon at which a word remains 90% retrievable.

On activation (speaking or hearing a word), stability is bumped up via a non-trivial SuperMemo-derived formula (`numba_comp_delta_S`, line 16):

$$\Delta S = \text{ds\_factor} \cdot \left( \text{act\_c} \cdot S^{-b} \cdot R \cdot a \cdot e^{c \cdot 100 \cdot d} + g \right)$$

with `a=7.6, b=0.023, c=-0.031, d=-0.2`. After the bump, `t` resets to 0; for *un*activated words, `t += 1` per step (`agent.py:753`), so their `R` decays exponentially without bound.

The `ds_factor` term is the key asymmetry: **speaking uses ds_factor = 1.0, listening uses 0.1** (`agent.py:546, 551`). Production reinforces memory ~10× more than comprehension — a sound empirical finding in SLA research.

### Thresholds

- `lang_act_thresh = 0.1` — agent must know ≥10% of vocabulary to *speak* a language
- `lang_passive_thresh = 0.025` — ≥2.5% to *understand*
- `pct_threshold = 0.9` — per-word "fully known" cutoff
- `switch_threshold = 0.2` — below 20%, agent gets reclassified as monolingual

## 3. Speaking and Listening Dynamics

### `pick_vocab` — three-stage sampling

When an agent speaks, `pick_vocab` (`agent.py:853-957`) executes three steps:

1. **Conceptual draw.** Sample `n` word indices from the age-appropriate Zipf CDF via `randZipf`. This represents the speaker's *intent* — what concepts they want to express — and is language-agnostic.
2. **Variant selection.** Given a chosen language (L1 or L2), pick pure vs. mixed store by comparing `pct`: if `pct(L1) ≥ pct(L2)` use pure L1, else fall back to L12. Bilinguals drift toward their dominant variant.
3. **Retrievability filter (the actually-spoken set).** For each conceptual word, draw a uniform random in [0,1]; the word is uttered only if the draw ≤ R for that word. Failed retrievals **cascade** to the next store: L1 → L12 → L2 (or symmetric). This cascade is exactly where code-switching emerges as an *emergent* phenomenon, not a programmed one.

### Listening and unknown-word acquisition

`process_unknown_words` (`agent.py:557-642`) handles two cases:

- **Cognate recognition** (lines 589-608): If a heard word in the other language is "close enough" — Levenshtein edit distance < `max_edit_dist=2`, with distances pre-drawn from a binomial(10, 0.3) at init — its `wc` is incremented in the home language too. This models cross-linguistic transfer (e.g. a Spanish speaker recognising Italian *acqua*).
- **Probabilistic acquisition** (lines 610-640): Genuinely unknown words enter the listener's `wc` array with probability `max(pct_understood, 0.1)` — i.e. the more of the conversation you already understood, the more likely you pick up a new word from context. Words only enter `S`/`R` once their `wc` crosses the per-word `mem_eff` threshold — a biologically-motivated **encoding threshold** (you need to hear something 3-7 times before it sticks).

## 4. Code-Switching: An Emergent Property

This is the model's most interesting design choice. **Code-switching is not programmed; it falls out of the retrieval cascade.** A bilingual with weak L2 reaches for an L2 word, fails the R-threshold roll, and the algorithm patches the gap from L12 or L1. The output utterance is a mosaic of stores — exactly what real bilingual speech looks like ("Voy a la *grocery store*").

The `L12`/`L21` arrays serve as a *bridging layer* that accumulates the lexicon of "things I usually express in L1 even when speaking L2". They are not initialised with anything (S=0.01, t=1000 — essentially null) and only grow through fallback events.

## 5. Conversation Language Choice: MAXIMIN

`get_conv_params` (`model.py:303-443`) implements Van Parijs's **MAXIMIN principle**: pick the language that *maximises the minimum* comprehension across participants. Decision tree:

1. All monolinguals on the same side → that language wins.
2. Bilingual(s) + monolingual(s) on one side → monolingual's language wins (the bilingual accommodates).
3. All bilinguals → each speaker can use their own dominant language (multilingual conversation).
4. Hard monolinguals on incompatible sides → conversation collapses to length=1 and one agent is *muted* (listens only).

When agents have a shared history (`known_people_network`), the previously-agreed language is reused — modelling the well-known stickiness of relationship-specific language choice.

Conversation length itself is age-modulated by `conv_length_age_factor` (lines 165-200): a sigmoid ramp from 0→1 by age 14, plateau through 65, then exponential decay. This is a developmental capacity proxy, not a learning-rate proxy.

## 6. How Bilingualism Emerges

Three channels feed L2 into an initially monolingual child:

1. **Family network.** `get_newborn_lang` (`model.py:217-249`) assigns the newborn's category from parental profiles. Children of one-L1 / one-L2 parents are automatically bilingual (category=1) and receive HDF5-loaded initial L2 vocabulary scaled by a `biling_key` ∈ {10, 25, 50, 75, 90} — five degrees of intergenerational transmission.
2. **School.** Every school has `lang_policy = [0, 1]` (`city_objects.py:80`) — meaning teachers may be drawn from either language. Children pick up L2 words through teacher-led and peer conversations governed by the MAXIMIN rule.
3. **Friendship network.** Friendships only form when language overlap is < 30% apart (`agent.py:1075`) and the pair has met ≥10 times — so the friendship graph itself is shaped by linguistic similarity, creating feedback loops.

**Reverse process — forgetting.** If an adult stops using a language, `t` increments unchecked, `R` decays exponentially, and once `pct < switch_threshold=0.2`, `update_lang_switch` (lines 265-288) reclassifies them as monolingual. There's also a reactive loop: when an agent is *linguistically excluded* (muted) in conversations repeatedly, `react_to_lang_exclusion` triggers `study_vocab` — a self-directed L2 study event. This is the closest thing to motivated learning in the model.

## 7. Quality Assessment

### What's done well

- **Ebbinghaus / SM-2 grounding** is psychometrically defensible. The exponential forgetting curve and stability-based consolidation are well-supported in cognitive science.
- **Production-comprehension asymmetry (10:1)** captures a real empirical finding rather than glossing over it.
- **Code-switching as emergence** is elegant — no separate "switching engine", it just falls out of the retrieval cascade.
- **Per-word memory state** (not aggregate proficiency) allows the model to express things a scalar proficiency can't: e.g. a bilingual who knows kitchen vocabulary in L1 only and work vocabulary in L2 only.
- **Three-stage Zipf-Mandelbrot** is a more honest fit to real word-frequency curves than a single power law.
- **MAXIMIN with conversation history** correctly captures the "we always speak X together" stickiness real bilingual dyads exhibit.

### Weaknesses & questionable choices

1. **Known bug in L12/L21 percentage update** — `pct` for `L12` is documented as never updated (`agent.py:916`), which breaks the dominance comparison at `pick_vocab` step 2. The intended logic silently degrades.
2. **L12/L21 never receive direct learning.** They only grow as fallback artefacts. Real bilinguals can study code-switched registers explicitly (e.g. Spanglish slang); the model has no path for that.
3. **Edit distance is a poor cognacy proxy.** Levenshtein between *opaque token indices* is geometric, not semantic. Two truly cognate words may sit at orthographically distant indices; two unrelated words at adjacent indices will be falsely "recognised". The `max_edit_dist=2` threshold is arbitrary.
4. **Age-invariant memory parameters.** The SM-2 constants `a, b, c, d` are global. There is no critical-period effect: a 2-year-old and a 30-year-old have identical consolidation rates. This contradicts robust SLA findings (sensitive period for phonology and morphology).
5. **Workplace has no language policy** by default. In real bilingual societies, the labour market is often the strongest L2-shift driver (e.g. Catalan/Spanish in Catalonia, French/English in Quebec). Treating jobs as language-neutral leaks the most important policy lever.
6. **Hard-coded school policy `[0, 1]`** — every school is bilingual. Real-world variation (immersion schools, dominant-language schools, heritage-language schools) is unrepresentable without code changes.
7. **Fixed home language.** Once a family's home language is set at birth, it doesn't shift over the family's lifetime — but real families' home languages do drift, especially in 2nd/3rd generation immigration scenarios.
8. **Magic constants everywhere** (`max_edit_dist=2`, `mean_word_distance=0.3`, `mem_eff ∈ [3,7]`, `biling_key` quintile, `switch_threshold=0.2`, etc.) without sensitivity analysis or sourcing in the code.
9. **No phonology / morphology / syntax layers.** Languages are bags of word-frequency tokens. This is fine for vocabulary-level shift dynamics but invisible to phenomena like accent retention, grammatical interference, or syntactic borrowing.
10. **Exclusion reaction is feeble.** When repeatedly muted, an agent studies *one* word (line 1035). A real human would dramatically restructure exposure (move, change friends, take a class).
11. **Vocab compression is static.** 500/1000 slots is a fixed budget. A child has the same array length as an adult, scaled only by which Zipf ranks they can sample. Real lexicon size grows by orders of magnitude over childhood.

## 8. Suggestions for Improvement

In rough order of impact:

1. **Fix the L12 `pct` bug** — without it the dominance logic is silently broken. This is a one-line fix.
2. **Add age-dependent consolidation.** Multiply the SM-2 `a` parameter by a sigmoid factor of age — high for ages 0–12, declining to ~0.6 for adults, ~0.3 for elderly. Even a coarse implementation would let the model exhibit a critical-period effect.
3. **Make workplace language policy configurable** and parameterise it per cluster/sector. This is the single biggest realism win for European-style bilingualism modelling.
4. **Heterogeneous school policies.** Allow `lang_policy ∈ {[0], [1], [0,1], weighted}` and let policy vary by cluster. Enables scenarios like immersion enclaves vs assimilationist regions.
5. **Replace edit-distance cognacy with a curated overlap matrix.** Pre-compute (offline) a sparse cognate map between L1 and L2 indices and look up rather than computing Levenshtein on opaque indices. Or at minimum, use *typological distance* as a global scalar gating recognition.
6. **Strengthen `react_to_lang_exclusion`** — make it scale with exclusion intensity, and let it bias the agent's social network rewiring (befriend more L2-dominant peers).
7. **Add a `home_lang_drift` mechanism** so families can shift home language across generations as adult proficiency profiles change.
8. **Sensitivity analysis harness.** Add a small script that sweeps the magic constants (`max_edit_dist`, `switch_threshold`, `mem_eff` range, `biling_key`) and reports macro-outcome variance. The model has too many free parameters to trust any single run.
9. **Expose schooling intensity / contact hours** as a parameter. Currently every school day generates one conversation; in reality, intensive immersion vs. background exposure differ by an order of magnitude in contact hours.
10. **Add active vs. passive bilingual distinction at output**. Right now anyone with `pct(L1), pct(L2) > 0.1` is "bilingual", but the production/comprehension asymmetry (built into the model!) means agents may be receptive-only bilinguals. Surface this in the macro-level classification.
11. **Make code-switching learnable.** Allow explicit reinforcement of `L12`/`L21` entries via repeated successful fallback events, so a stable "third dialect" can emerge in long-running mixed populations — which is what actually happens in real bilingual communities (Spanglish, Franglais).
12. **Vocabulary growth.** Replace the static `vocab_red` with an age-indexed *active* size; or, more elegantly, treat `R` near 0 as "word not yet present in the agent's lexicon" and let the ceiling grow naturally as the age-CDF widens.
