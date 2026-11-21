# Review of the learning-model documents: shortcomings and inconsistencies

> **Status (Nov 2026):** review only. Neither reviewed document has been
> changed. This review covers two documents:
> - [`memory_model_analysis.md`](memory_model_analysis.md), cited here as
>   **MMA** (its findings are M1–M13);
> - [`learning_model_new_approaches.md`](learning_model_new_approaches.md),
>   cited as **LMNA** (its shortcomings are S1–S17, its validation targets
>   V1–V18).
>
> Findings are numbered **X** (inconsistencies between the two documents),
> **D** (MMA) and **N** (LMNA). Line numbers refer to the versions at commit
> `72b86c2`. Severity:
> - **High**: changes a recommendation, or makes a planned test or build step
>   fail.
> - **Medium**: changes how a result or an estimate should be read.
> - **Low**: a wording or number slip.
>
> Most numbers in both documents were re-checked and are correct
> ([Appendix C](#appendix-c--numbers-re-checked)). The new numbers in this
> review come from two small checks, described in
> [Appendix A](#appendix-a--act-r-check-time-unit-and-long-term-forgetting) and
> [Appendix B](#appendix-b--cost-of-approach-c-when-memory-bandwidth-is-the-limit).

---

## 0. Summary

The most important findings:

| ID | Finding | Severity |
|---|---|---|
| **N1** | The ACT-R rule proposed for A0 diverges with the time unit LMNA specifies (days). | High |
| **N2** | A power-law rule has no "permastore" plateau, yet LMNA lists it as built in. | High |
| **N8** | A's lexicon is closed: no synonyms, no new words, no integration of loanwords. | High (for the research question) |
| **N11** | A0's 5,000-lemma lexicon cannot run several tests in its own battery. | High |
| **X1** | The two documents recommend different memory rules, and LMNA does not say why it changed. | High |
| **X2** | The documents make contradictory assumptions about how age affects vocabulary encoding. | Medium–High |
| **X3** | LMNA's new threshold mapping contradicts MMA's own coverage equivalences. | Medium–High |
| **N20** | Initial conditions by spin-up, and the census target V17, need cohort-specific language history that the plan does not include. | Medium–High |
| **N26** | C0 lacks the child-directed text it needs. | High (for C) |
| **D1** | "Shift speed is likely over-estimated" is asserted; the net direction of the bias is unknown. | Medium |
| **N25** | C's cost may be under-estimated by up to about 10×. | Medium |

**Overall.** The diagnosis of the legacy model (MMA, and LMNA §2) is sound, and
its numbers hold up. The weak points are in the proposal:
- the memory rule chosen for A0 is under-specified in ways that change its
  behaviour;
- several claims for A are stronger than its mechanism supports;
- the A0 lexicon is too small for A0's own tests;
- the representation cannot express vocabulary change *within* a language,
  which matters for a study of minority-language sustainability.

None of these overturns the recommendation to build the shared foundations and
A0 first. They change what A0 must contain, and the order in which decisions
should be made (§5).

---

## 1. Inconsistencies between the two documents

### X1. The recommended memory rule changes without explanation — *High*

- **Where:** MMA recommends an FSRS-style rule (C1), with ACT-R (C2) as the
  alternative (MMA:58-63, MMA:853-857). LMNA builds A0 on ACT-R with
  Pavlik–Anderson decay, and demotes FSRS and the multiscale context model to
  fallbacks (LMNA:574-576).
- **Problem:** LMNA does not explain the change, and it does not answer the
  drawbacks of ACT-R that MMA lists:
  - there is no lapse mechanism;
  - plain base-level learning does not reproduce the spacing effect;
  - activation-dependent decay in principle needs each word's history of uses
    (MMA:838-843).

  LMNA's reasons are implicit: ACT-R is grounded in the statistics of the
  environment (Anderson & Schooler, 1991), and in ACT-R every encounter adds a
  trace, which suits incidental exposure.
- **Fix:** LMNA should state why it prefers ACT-R and answer MMA's drawbacks.
  Better still, leave the choice to decision point D1, with all candidates
  compared on the harness (see N4). MMA's recommendation should be marked as
  superseded for the new learning core.

### X2. Contradictory assumptions about age and encoding speed — *Medium–High*

- **Where:** MMA M6 (MMA:392-394) cites evidence that adults learn vocabulary
  faster at first (Snow & Hoefnagel-Höhle, 1978). LMNA §4.4 (LMNA:583) assumes
  an encoding gain `g_enc(age)` that is "higher in childhood", and describes
  childhood memory as "fast-in/fast-out".
- **Problem:** The usual summary of the literature is "older is faster, younger
  is better in the long run" (Krashen, Long & Scarcella, 1979). The child
  advantage concerns ultimate attainment, mostly in phonology and grammar.
  Faster encoding of vocabulary per encounter in children is not established.
- **Fix:** Estimate `g_enc(age)` with a prior that allows either direction. Use
  separate targets for learning *rate* (e.g. Snow & Hoefnagel-Höhle) and for
  ultimate *attainment*.

### X3. The new threshold mapping contradicts MMA's coverage equivalences — *Medium–High*

- **Where:** MMA M7 (MMA:417-425) shows that, if an agent knows the most
  frequent slots, the legacy thresholds correspond to these coverages
  (coverage = share of running words known):

  | Legacy threshold | Role | Coverage |
  |---|---|---|
  | 0.025 | "understands" | ≈ 79 % |
  | 0.1 | "can speak" | ≈ 92 % |
  | 0.2 | bilingual label | ≈ 95.5 % |

  LMNA (LMNA:307-312) maps `lang_passive_thresh = 0.025` ("understands") to
  "expected coverage of everyday speech ≥ ~95 %".
- **Problem:** LMNA's "understands" is MMA's *bilingual* threshold, a much
  stricter test than the legacy one. MAXIMIN would then classify many more
  agents as real monolinguals, which changes macro dynamics before any learning
  effect. The mapping may be intended, but it is presented as a translation of
  the legacy thresholds, not as a re-definition.
- **Fix:** Label the mapping as a re-definition. Run the legacy model with both
  the 79 % and the 95 % mappings, to measure how much the choice alone moves
  the macro outcomes.

### X4. M3 is framed around FSRS lapses, and "fixes M3" is only half true — *Medium*

- **Where:** MMA M3 (MMA:285-307) states the problem as "retrieval failures have
  no cost". It argues from the FSRS lapse rule. LMNA (LMNA:546-549) says ACT-R
  "fixes M3".
- **Problem:**
  - The FSRS lapse rule is a fitted description of stability after a failed
    review. It does not show that a failed retrieval damages memory. In
    laboratory studies, unsuccessful retrieval followed by exposure to the
    answer helps later learning (Kornell, Hays & Bjork, 2009).
  - The real defect in legacy is that `S` never decreases. A forgotten word
    keeps its old `S` and then receives the largest gain when it is met again.
  - ACT-R removes that defect: activation decays, and a new encounter adds a
    new trace. But it has no cost for failed retrieval either. So "fixes M3" is
    true of the defect, not of MMA's definition of it.
- **Fix:** In MMA, reword M3 as "stability never decreases, and relearning a
  forgotten word is rewarded". In LMNA, say which part of M3 is fixed.

### X5. "SM-2" is the wrong name for the legacy formula — *Low*

- **Where:** LMNA:189 ("the SM-2 `a, b, c, d`"). CLAUDE.md lines 7, 45 and 95
  use the same name.
- **Problem:** SM-2 (1987) schedules reviews with an easiness factor and has no
  stability function. MMA M1 correctly identifies the source as SuperMemo's
  stability-increase function `SInc(S, R)`, from the two-component model of
  memory (Wozniak, Gorzelańczyk & Murakowski, 1995).
- **Fix:** Use "SuperMemo stability-increase function" in LMNA and CLAUDE.md.

### X6. Should the legacy learner be fixed or frozen? — *Medium*

- **Where:** LMNA §2.5 (LMNA:234-240) says MMA's Options A–C "are worth
  applying to the legacy learner if it is kept as a baseline". LMNA §3.1 rule 4
  (LMNA:314-317) says "the legacy learning code stays unchanged".
- **Fix:** Define two plug-ins: `legacy` (verbatim, the reproducible
  reference) and `legacy-corrected` (MMA Option A). Comparing them separates
  the bookkeeping artefacts from the deeper representational problems.

### X7. Two roadmaps, and MMA does not point to its successor — *Medium*

- **Where:** MMA §4.7 (MMA:888-896) proposes this order: memory module, then
  Option A, then FSRS, then structural changes inside the slot model. LMNA §7.2
  proposes a new learning core. MMA's status block (MMA:15-20) does not mention
  LMNA.
- **Fix:** Add a note to MMA's status block: the diagnosis (§1–§3) stands; the
  improvement options (§4) now apply only to the `legacy-corrected` plug-in;
  the new learning core is described in LMNA.

### X8. Validation IDs are reused with changed content — *Low*

- **Where:** MMA's V-table (MMA:871-886) vs LMNA's (LMNA:403-422):
  - V6 is "frequent words are learned earlier and kept longer" (source
    "general") in MMA, but "known by more people and recognised faster" in
    LMNA;
  - V7 and V9 have different sources in the two documents.
- **Fix:** Declare LMNA's list canonical and have MMA refer to it.

---

## 2. Shortcomings of `memory_model_analysis.md`

### D1. "Shift is likely over-estimated" is asserted; the net bias is unknown — *Medium*

- **Where:** MMA:380-383 (impact of M5); repeated in LMNA §2.4 (LMNA:224-225).
- **Problem:** Adult attrition that is too fast (M5 with M7) does push toward
  over-estimating shift. But M2, M12 and M13 make learning far too fast: a
  quarter of the initial `L2` monolinguals became bilingual within 4 months
  (M13). In Catalonia, children of Spanish-speaking families and immigrants
  learn Catalan largely at school. Learning at school that is too fast would
  therefore *under*-estimate shift away from Catalan. The net direction is
  unknown.
- **Fix:** State both directions. Measure the net effect by comparing `legacy`
  and `legacy-corrected` on the same scenarios.

### D2. The coverage thresholds come from reading studies — *Medium*

- **Where:** M7 (MMA:411-415) and V7 (MMA:883) cite Hu & Nation (2000) and
  Laufer & Ravenhorst-Kalovski (2010).
- **Problem:**
  - Both are reading studies. Listening needs less coverage: about 95 % is
    adequate, and some listeners manage at 90 % (van Zeeland & Schmitt, 2013,
    which LMNA cites but MMA does not).
  - The norms are measured in word families, on L2 learners of English. A
    compressed slot is not a word family.
- **Fix:** Cite the listening evidence, and note the mismatch of units.

### D3. V5 ("childhood L1 loss can be near-complete") is overstated — *Medium*

- **Where:** V5 (MMA:881), M5 (MMA:353-355); also LMNA V5 (LMNA:409).
- **Problem:** Later adoptee studies find residual traces and faster relearning
  of a "lost" first language:
  - Pierce et al. (2014): adoptees with no conscious knowledge of Chinese still
    process Chinese tones neurally like native speakers;
  - Choi, Broersma & Cutler (2017): Korean adoptees adopted in infancy relearn
    Korean sound contrasts faster;
  - Bowers, Mattys & Gage (2009): adults relearn Hindi or Zulu sound contrasts
    heard in childhood faster than controls.
- **Why it matters:** The target decides between models. Legacy's residual
  stability and ACT-R's old traces can both produce relearning savings. A
  target of "complete loss" would wrongly penalise that behaviour.
- **Fix:** Restate V5 in both documents as "accessible knowledge (recall,
  comprehension) collapses; relearning savings remain".

### D4. M5's severity largely belongs to M7 — *Low–Medium*

- **Where:** M5 is rated High (MMA:43). Its own example (MMA:362-369) shows
  that after 40 years without use, the adult still recalls 93 % of running
  words.
- **Problem:** Frequent words have stabilities of centuries (M1 table), so the
  exponential shape matters mostly for mid-frequency and rare words. The harm
  shown in M5 runs mainly through `pct` (M7).
- **Fix:** Make M5's dependence on M7 explicit, and consider rating M5 Medium.

### D5. The "speaking" examples for new words cannot happen through speech — *Low*

- **Where:** §1.4 step 5 (MMA:168-170) and the speaking column of the M2 table
  (MMA:257-262).
- **Problem:** A new word has `R ≈ 0`. `pick_vocab` utters a word only with
  probability `R`, so a new word cannot be spoken. Speaking-strength updates of
  a new word happen only through `study_vocab` (`ds = 1`) or the code-switch
  cascade into `L12` (M4).
- **Fix:** Relabel these cases as studying or as the code-switch cascade.

### D6. The "one per step" row mixes two problems — *Low*

- **Where:** §1.4 step 4 (MMA:164-165) and the M2 table (MMA:260).
- **Problem:** Because of the off-by-one in `t`, a word used every step always
  has `t = 0` and `R = 1` (M12). The low `S` in that row therefore reflects M12
  as much as the additive rule (M1).
- **Fix:** Say so, and show the row again with fix A3 applied.

### D7. M1's summary overstates the absence of a spacing benefit — *Low*

- **Where:** MMA:39 ("spaced repeated use adds almost nothing after the first
  encounter").
- **Problem:** The M2 table shows one use per year giving `S = 13.7`, against
  9.4 for one use per step: 46 % more. Legacy does reward long gaps, because the
  gain grows as `R` falls. What is wrong is that cramming beats spacing, and
  that each use adds a small fixed amount rather than multiplying `S`.
- **Fix:** Reword the summary row.

### D8. FSRS has no model for passive exposure — *Medium*

- **Where:** MMA §4.4 C1 (MMA:801-807): production counts as "Good", failed
  retrieval as "Again", and listening as a weaker review scaled by `ds`.
- **Problem:** FSRS is fitted on deliberate retrieval tests (flashcards). Most
  natural exposure is *hearing* a word, which is restudy, not a test, and FSRS
  has no model for it. Scaling the growth term by `ds` is ad hoc, and no data
  supports it.
- **Fix:** Name this as the main open problem of C1. It is also an argument
  for rules in which any encounter adds a trace (ACT-R, the multiscale context
  model).

### D9. The legacy vocabulary ceiling is not flagged — *Low–Medium*

- **Where:** M8 (MMA:447-458). The ceiling in adulthood is 10,000 words.
- **Problem:** That is about 4× below the ~42,000 lemmas known at 20 years
  (Brysbaert et al., 2016, cited by LMNA V9). The low ceiling affects M7 and
  any vocabulary-size comparison.
- **Fix:** Mention it in M8.

### D10. A runtime crash noted in an appendix is not tracked elsewhere — *Low*

- **Where:** Appendix A4 (MMA:951-957). A run crashed with `self.pos is None`,
  and the 10-year run behind M13 skipped 49 `listen()` calls.
- **Problem:** No plan or issue records the crash, and the skipped calls
  slightly bias the M13 numbers.
- **Fix:** Track the crash, and note the bias in M13.

### D11. Number and wording slips — *Low*

| Where | Text | Correction |
|---|---|---|
| MMA:302-304 | expected gain `R·g` "peaks at R ≈ 0.3–0.4 (0.73 steps)" | 0.77 at R ≈ 0.30 for S = 30 (0.75 for S = 100) |
| MMA:757 vs 763 | Option B's rule uses `S1` "e.g. 0.3 steps"; its worked numbers use `S1 = 1` | Make them consistent. The result barely changes: 15.7 vs 16.3 for one use per step. |
| MMA:40 | "15× more durable than 10 uses spread over a year" | The rows compared span 100 days (one per step) and about 400 days (one every 4 steps). "Spread over months" is accurate. |

---

## 3. Shortcomings of `learning_model_new_approaches.md`

### 3.1 The memory mechanism of approach A

#### N1. The proposed ACT-R rule diverges with the stated time unit — *High*

- **Where:**
  - §3.1 rule 1 (LMNA:286) measures time in days;
  - §4.3 (LMNA:566-569) spreads repeats within one day over waking hours, so
    many gaps between encounters are fractions of a day;
  - §4.3 (LMNA:551-555) leaves the time scaling open ("check its exact form in
    the 2005 paper").
- **Problem:** In the Pavlik–Anderson rule, a use's decay `d_j` grows with
  `exp(m)`, the activation at that use. When a gap is below 1 time unit,
  `gap^(−d)` *grows* with `d`. High activation then gives a larger decay, which
  gives a larger contribution from recent short gaps, which gives higher
  activation: runaway positive feedback.

  With the values commonly quoted from Pavlik & Anderson (2005), `c = 0.217` and
  `α = 0.177` (Appendix A):
  - a word used 3 times a day **diverges** (decay → ∞) when time is in days;
  - after use stops, that word's recall falls to 0.04 within a year;
  - with time in seconds, ACT-R's native unit, gaps are always above 1, the
    feedback is negative and the rule is stable.

  So the time unit is not a detail: it changes the model qualitatively.
- **Fix:**
  - Specify the unit (seconds), or a minimum gap.
  - Specify the scaling of time between sessions, and decide what a "session"
    is in natural exposure. Pavlik & Anderson's scaling applies between
    practice sessions, and daily life has no sessions.
  - Make both part of decision point D1.

#### N2. A power-law rule has no permastore plateau — *High*

- **Where:** §4.3 (LMNA:544-545) says a sum of power functions gives "a plateau
  for well-practised words". §7.1 (LMNA:1103) lists permastore as "built in"
  for A.
- **Problem:**
  - A power law has no plateau: activation keeps falling by about
    `d·ln(T₂/T₁)` between two delays `T₁` and `T₂`.
  - In the stable configuration (seconds), a word learned at school (twice a
    week for 3 years) is recalled with probability 0.61 after 5 years and 0.23
    after 25 years (Appendix A; the retrieval threshold and noise are
    illustrative). Bahrick (1984) found retention essentially flat between
    about 5 and 30 years.
  - Rescaling time by a constant cannot create a plateau, because it leaves
    the ratio `T₂/T₁` unchanged.
- **Fix:**
  - Replace "built in" with "to be fitted and tested".
  - A plateau needs a mechanism by which old, consolidated traces stop
    decaying. Options: a decay that falls with the age of the trace or with
    spacing, the slow time scales of the multiscale context model, or a
    two-component trace.
  - D1 already tests V3. Make V3 a gating criterion for the memory rule.

#### N3. The compact-state approximation compounds; V10 cannot be exact for the whole learner — *Medium*

- **Where:** §4.3 compact state (LMNA:559-572); A0 acceptance requires V10 to
  hold "exact, within floating-point tolerance" (LMNA:707).
- **Problem:**
  - Each use's decay depends on the activation at that use. That activation is
    computed from the compact state: the last K uses kept exactly, plus Petrov's
    summary of the older ones with a *mean* decay. Errors in activation
    therefore feed into every later decay. Over decades and millions of uses
    they compound.
  - Petrov's approximation assumes one common decay. Here the decays span a
    wide range.
  - Production, comprehension and meaning inference are random draws. The
    invariance tests can be exact only for the deterministic memory update,
    given a fixed list of events. For the full learner they hold only in
    distribution, unless random streams are keyed to each event.
- **Fix:** Restate V10: exact for the memory update given the events,
  statistical for the full learner. Bound the error of the compact state over
  full 80-year lifetimes, not only over short scenarios.

#### N4. Memory rules that update exactly without history are relegated to fallbacks — *Medium*

- **Where:** LMNA:574-576 (FSRS and the multiscale context model, only "if
  ACT-R fails the harness").
- **Problem:**
  - The multiscale context model (Mozer et al., 2009) keeps a few leaky
    integrators per word, at different time scales. It updates exactly and
    step by step, without storing past uses, and it produces both power-law-like
    forgetting and the spacing effect. That is exactly what R5 asks for, with
    no approximation error (cf. N3).
  - The Predictive Performance Equation (PPE) was designed for long retention
    intervals. Walsh et al. (2018) compared it with Pavlik–Anderson ACT-R on
    spacing data.
- **Fix:** At D1, compare ACT-R, the multiscale context model, FSRS and PPE
  from the start on V1–V5 and V10 and on cost, then choose.

#### N5. The "rusty L1" effect is claimed but not produced — *Medium*

- **Where:** §4.6 (LMNA:627-633), which cites Linck, Kroll & Sunderman (2009).
- **Problem:**
  - In A's competition formula, strong recent use of the other language `O`
    lowers the chance of *selecting* the target word `T`. That raises
    switching into `O`.
  - But the recall probability `P_prod(T)` and the latency `RT(T)` depend only
    on `T`'s own activation.
  - Linck et al. measured *reduced L1 fluency* (fewer L1 words produced in
    verbal fluency tasks): harder retrieval of L1, not more switching.
  - With a monolingual listener (large control bias `b`), A predicts no
    rustiness at all.
- **Fix:** Add language-level inhibition (Green, 1998): a per-agent,
  per-language control state, raised by use of the other language, that lowers
  access to the inhibited language (for example by raising its recall
  threshold). Alternatively, make response times depend on the competitor's
  activation. Calibrate on Linck et al.'s fluency data.

#### N6. The cognate effect enters three times — *Medium*

- **Where:**
  - item difficulty `β_w`, derived from cognate strength (LMNA:556-557);
  - the cognate recognition route `P_cog` (LMNA:598-603);
  - the probability of inferring a meaning, `p_inf`, which rises with cognate
    strength (LMNA:609-611).
- **Problem:** V14 (cognate naming latency) is the only cognate target. Three
  paths serving one target cannot be identified separately.
- **Fix:** Start with one path. Add another only when a target requires it.
  For example, use de Groot & Keijzer's learning and forgetting data to justify
  `β_w`.

#### N7. The speaking/listening asymmetry is grounded in the wrong effect — *Low*

- **Where:** LMNA:110-112 and LMNA:535-539, citing the production effect
  (MacLeod et al., 2010).
- **Problem:** The production effect compares reading aloud with reading
  silently while studying. Saying a word in conversation means retrieving it
  from memory: the testing effect (Roediger & Karpicke, 2006) and the
  generation effect (Slamecka & Graf, 1978). Both are larger, better studied,
  and come with retention data that could calibrate the weights for speaking
  (`ω_p`) and for understanding in context (`ω_c`).
- **Fix:** Cite them, and use their retention data as priors.

### 3.2 Representation

#### N8. The lexicon is closed: no synonyms, no new words, no integration of loanwords — *High (for the research question)*

- **Where:** §3.2 (LMNA:323-326): a concept is "a set of translation-equivalent
  lemmas, one per language at first", restricted to one-to-one equivalents
  (LMNA:679). The competition formula of §4.6 sets the target word against
  only its translation equivalent. The A1 "borrowing habit" (LMNA:644-647) is
  the only extension mentioned.
- **What cannot be represented:**
  - **Established loans.** *Vale* or *bueno* used inside Catalan become Catalan
    words for the people who use them. In A, a Catalan child who hears *vale*
    in Catalan speech reinforces the *Spanish* lemma. This misattributes
    knowledge, and it hides lexical convergence of the minority language toward
    the majority one. A does not distinguish code-switching from established
    borrowing (Poplack, Sankoff & Miller, 1988).
  - **Existing synonyms.** Dialect variants (*noi / xiquet / al·lot*), standard
    vs colloquial forms, and child-register words. The CDI checklists include
    baby words.
  - **Vocabulary turnover**: new terms from media and official terminology,
    replacement of one variant by another. Core words such as *pa* and *aigua*
    are among the most stable (Pagel, Atkinson & Meade, 2007). Contact and
    turnover act mostly on discourse markers and on mid-frequency, domain
    vocabulary.

  LMNA's comparison table (§7.1) does not list this as a limit of A. Only C,
  with its open-ended output, could produce such variation, and there it is
  hard to measure.
- **Fix (a variant layer, cheap to add to A):**
  - **Inventory.** For each concept, a set of variants per language (native,
    dialectal, normative, loan), drawn from aligned Spanish and Catalan
    WordNets (the Multilingual Central Repository) plus lists of known
    Castilianisms.
  - **Memory.** Each variant has its own trace, plus a learned, per-agent
    weight for "which language this variant belongs to", updated from the
    language of the speech it is heard in.
  - **Production.** A choice among all the variants of the concept, weighted by
    memory strength, the bias toward the intended language, accommodation to the
    listener, and normative pressure from school and media. Near-synonyms
    compete in production much as translation equivalents do (Peterson & Savoy,
    1998).
  - **New variants** arrive from outside (media, official new terms), or by
    conversion: a code-switched lemma used often enough inside `L` speech
    becomes an `L` variant for that agent.
  - **Model and tests.** This is the utterance-selection model of language
    change (Baxter, Blythe, Croft & McKane, 2006). It predicts the S-shaped
    spread of new variants (Blythe & Croft, 2012). It can be tested against
    rates of Castilianisms by age and language dominance in spoken corpora.
  - **Cost:** about 3–5 extra parameters, and about 1.2–2× the memory.
    Coining genuinely new forms would still need C.

#### N9. The one-to-one filter drops the most frequent, most contact-sensitive words — *Medium*

- **Where:** LMNA:324-326 and LMNA:679 (lemmas present in both frequency lists,
  one-to-one equivalents only).
- **Problem:** The filter drops or distorts:
  - function words without a one-to-one equivalent;
  - the Catalan pronominal clitics *en* and *hi*;
  - discourse markers;
  - baby words;
  - concepts specific to one language.

  These are among the most frequent tokens, so coverage computed on what
  remains is biased. They are also the items most affected by contact.
- **Fix:** Use a many-to-many concept table, entries for concepts specific to
  one language, and a separate class for function words that counts toward
  coverage.

#### N10. The role of embeddings in A is overstated and not justified — *Low*

- **Where:** The summary table (LMNA:66-74) says that in A, embeddings "give
  meaning, semantic fields and context (from stage A1)".
- **Problem:**
  - In A, embeddings never give agents meaning. They are a fixed resource,
    shared by all agents, used for two things: grouping concepts into topic
    fields so that what is said depends on the setting, and measuring how
    similar an unknown word's concept is to its context when inferring its
    meaning.
  - A0 uses no embeddings.
  - Both A1 uses could be done without embeddings: word-frequency lists
    computed per setting from register corpora, and co-occurrence counts. These
    are easier to test against data and add fewer free parameters (no number
    of fields, no affinity function).
- **Fix:** State plainly that A's knowledge is never stored as embeddings, and
  that A0 uses none. Justify embeddings over per-setting frequency lists, or
  drop them.

### 3.3 Tests, lexicon size and calibration

#### N11. A0's lexicon is too small for A0's own tests — *High*

- **Where:** A0 keeps the top 5,000 lemmas per language (LMNA:677). The battery
  includes vocabulary size measured over about 40,000 lemmas (LMNA:385), adult
  V9 (about 42,000 lemmas at 20 years, LMNA:413), and LexTALE-Esp (LMNA:386,
  LMNA:690-692).
- **Problem:**
  - Adult vocabulary size cannot exceed 5,000 per language.
  - At the assumed 25,000 tokens a day (Appendix A of LMNA), an adult meets
    most of the top 5,000 lemmas constantly. Differences between adults
    saturate.
  - LexTALE discriminates through low-frequency items, which are partly
    outside the top 5,000.
  - B0 reports tokens outside its lexicon separately (LMNA:870-874); A0 has no
    equivalent for coverage.
- **Fix:** Either restrict A0's tests to what 5,000 lemmas can support (CDI,
  vocabulary in childhood, coverage with out-of-lexicon tokens reported
  explicitly), or enlarge A0 to 20,000–30,000 lemmas per language with sparse
  storage. At single-agent scale the cost is memory, not time.

#### N12. Lexical-decision and prevalence tests need a model for pseudowords — *Medium*

- **Where:** A0's battery: simulated LexTALE-Esp and vocabulary size
  (LMNA:690-692).
- **Problem:** Both tests ask "is this a word / do you know it?" about real
  words and pseudowords. A has no representation of word forms beyond cognate
  strength, so it cannot produce false alarms.
- **Fix:** Add a response model: signal detection with a response bias, and
  false alarms that rise with a pseudoword's similarity to known lemmas.

#### N13. Targets held out for validation are also used for fitting — *Medium*

- **Where:** §3.6 item 5 (LMNA:435-438) holds out attrition and Thordardottir's
  exposure curves. §4.4 (LMNA:591-592) calibrates the age curves on V4, V5, V9
  and V13; V13 comes from Thordardottir and Unsworth.
- **Fix:** Choose one split. For example, fit the age curves on V9 and V5, and
  hold out V4 and V11–V13.

#### N14. The set of harness targets differs between sections — *Low*

- **Where:**
  - the harness covers V1–V15 (LMNA:693);
  - the recommended path says V1–V16 (LMNA:1125);
  - calibration by simulation-based inference uses V1–V16 (LMNA:431);
  - A0's acceptance list leaves out V7 and V15 (LMNA:703-712).
- **Fix:** Use one list everywhere.

#### N15. Simulation-based inference is the wrong tool for qualitative targets — *Medium*

- **Where:** §3.6 (LMNA:426-439).
- **Problem:**
  - V1–V3, V8 and V12–V14 are orderings or curve shapes, taken from studies in
    other languages and paradigms.
  - Neural posterior estimation needs summary statistics and an observation
    model.
  - Putting qualitative constraints and quantitative data into one posterior
    is ill-defined.
- **Fix:** Calibrate in two stages:
  1. **History matching** with emulators (Vernon, Goldstein & Bower, 2010;
     Andrianakis et al., 2015) rules out the parameter regions that violate
     the qualitative targets and the pre-set acceptance bands.
  2. **Simulation-based inference** then uses the quantitative targets (CDI,
     prevalence norms, LexTALE, census), each with an explicit model of
     measurement error.

#### N16. A's number of free parameters is understated — *Medium*

- **Where:** §7.1 (LMNA:1111) gives "~15–20, interpretable".
- **Problem:** A0 as specified has:
  - the decay constants `c` and `α`;
  - two age curves with 2–3 parameters each;
  - the recognition and recall thresholds `τ_comp` and `τ_prod`, and the noise
    `s`;
  - the response-time parameters `F` and `t0`;
  - three event weights;
  - a meaning-inference function with 2–3 parameters;
  - the cognate exponent `κ`;
  - the language-control bias `b` and how it depends on the listener;
  - the cap on repeats within a day;
  - the time scaling;
  - the old-age threshold curve;
  - the variance of the forgetting-rate trait;
  - the coverage threshold `θ_cov`.

  That is about 25–35, before counting the exposure model and the mapping to
  self-reports.
- **Fix:** Use the honest count when judging R6 (identifiability).

#### N17. Invariance to the step length holds for the learner, not for the coupled model — *Medium*

- **Where:** §3.1 rule 1 (LMNA:286-290): "results do not depend on the ABM's
  10-day step".
- **Problem:** That is true for the learner, given a list of events. But the
  social layer still generates events step by step (who meets whom, `num_days`),
  and it must assign them to days by some rule. The step length and that rule
  still shape the stream of events.
- **Fix:** Claim invariance at the learner level. Test the coupled model's
  sensitivity to the step length separately. Specify the rule for assigning
  days (weekdays for school and work, and so on).

### 3.4 Exposure model and initial conditions

#### N18. The exposure model ignores who is speaking — *Medium*

- **Where:** §3.3 (LMNA:340-374).
- **Problem:**
  - LENA adult word counts include overheard speech. Weisleder & Fernald
    (2013) found that speech directed to the child, not overheard speech,
    predicted vocabulary growth.
  - Place & Hoff (2011) found that the share of input coming from native
    speakers matters. This is directly relevant to Catalonia, where many
    parents speak to their children in their second language.
- **Fix:** Split input into directed vs overheard and native vs non-native, with
  weights to be estimated.

#### N19. Poisson word sampling under-represents massing; contextual diversity is unused — *Medium*

- **Where:** §4.6 (LMNA:619-621), which draws each concept's count from a
  Poisson distribution.
- **Problem:**
  - Real word use is bursty: once a topic word appears, it tends to recur
    (Church, 2000; Altmann, Pierrehumbert & Motter, 2009). Burstiness decides
    how massed exposure is, and the memory rule is built to respond to
    spacing. A Poisson draw therefore biases exactly the quantity the rule is
    sensitive to.
  - Contextual diversity predicts word recognition better than frequency
    (Adelman, Brown & Quesada, 2006). SUBTLEX-CAT and SUBTLEX-ESP already
    provide it.
- **Fix:** Sample words conditioned on topic, or from a negative binomial. Add
  contextual diversity to V6.

#### N20. Spin-up needs cohort-specific language history; V17 needs real demography — *Medium–High*

- **Where:** §3.7 (LMNA:441-454) spins up each initial agent from "its language
  type, family and schooling". V17 compares with census and EULP knowledge by
  age, birthplace and years of residence (LMNA:421).
- **Problem:** In Catalonia, cohort history dominates knowledge by age:
  - Catalan was excluded from schools and public life in 1939–75;
  - the 1983 language-normalisation law and immersion schooling followed;
  - large immigration from the rest of Spain arrived in the 1950s–70s.

  The census pattern by age reflects this history; for example, older cohorts
  who speak Catalan but cannot write it. Without exposure histories specific
  to each birth cohort, and a social layer with real demography, V17 cannot
  pass. Decision point D2 ("check the exposure model before the learner")
  would then always fire.
- **Fix:** Add historical exposure scenarios for each birth cohort to the
  exposure model. State explicitly that V17 depends on demography and history,
  which the social layer does not yet model.

### 3.5 Resource estimates

#### N21. The summary's cost for A covers only the learning core — *Medium*

- **Where:** The summary table (LMNA:73) gives A at about 2–4 h per run for
  10,000 agents. §4.11 (LMNA:751-754) notes that the social layer today takes
  about a third of the runtime, roughly 20 of the 60 hours.
- **Fix:** Show both figures in the summary.

#### N22. The memory per trace is under-estimated — *Low*

- **Where:** "About 24 bytes" (LMNA:517, LMNA:572, LMNA:1188).
- **Problem:** The described state holds 2–3 recent uses (time and decay,
  8 bytes each in float32) plus a summary (count, first use, last use, mean
  decay: 16 bytes). That is 32–40 bytes. A0 then needs 3.2–4 GB at 10,000
  agents, not 2.4, and A4 needs 19–24 GB, not 14.4. Both still fit the planned
  hardware.
- **Fix:** Correct the figures.

#### N23. "~1 MB per agent lifetime" does not match "2,000 agents ≈ 0.5 GB" — *Low*

- **Where:** LMNA:747. The 0.5 GB matches 240 kB of state per agent, not 1 MB.
- **Fix:** Say what the 1 MB refers to.

#### N24. Proficiency assessments are not costed — *Low*

- **Where:** §3.1 refreshes proficiency monthly through `assess`. Appendix A of
  LMNA does not include it.
- **Problem:** A monthly assessment evaluates every lemma of every agent:
  10,000 × 10,000 × 1,200 months ≈ 1.2 × 10¹¹ evaluations, about 10 % of the
  learning updates.
- **Fix:** Add it to the estimate.

#### N25. C's cost may be under-estimated by up to about 10× — *Medium*

- **Where:** Appendix A of LMNA (LMNA:1223-1240) assumes that throughput is
  limited by computation, at 25 % of peak.
- **Problem:** Each agent has its own weights. Unless work is batched per
  agent, both generation and per-utterance SGD are limited by memory bandwidth,
  not computation (Appendix B):
  - generation: up to about 7,300 GPU-hours, not about 100;
  - training with one SGD step per 20-token utterance: about 2,700 GPU-hours,
    not about 730;
  - replay is not costed at all.

  In the worst case that is about 10,000 GPU-hours against 830. Batching each
  agent's work over a whole day restores the original estimate, but it changes
  the learning dynamics (one large update per day instead of one per
  utterance), and the social layer must then plan each day's conversations in
  advance.
- **Fix:** Show both cases, cost replay, and state the batching assumption.

### 3.6 Approaches B and C

#### N26. C0 lacks the child-directed text it needs — *High (for C)*

- **Where:** §6.6 (LMNA:1017-1031); CHILDES is flagged as "small"
  (LMNA:331).
- **Problem:** Each C0 agent processes about 20,000 tokens a day for 10,950
  days, about 2.2 × 10⁸ tokens by age 30. Its first five years of child-directed
  input alone come to more than 10⁷ tokens. The Catalan and Spanish CHILDES
  corpora are far smaller than that. Agents would loop over the same small
  corpus many times. That brings memorisation and artificially low surprisal on
  repeated text, and the 60 agents would not be independent samples. Catalan
  subtitle corpora are also limited.
- **Fix:** Measure the corpus size per register before C0. Generate
  child-directed text with a teacher model (with the caveat in §6.9), or
  restrict C0 to ages and registers that have enough data.

#### N27. Plain SGD is a risky choice for transformers — *Medium (for C)*

- **Where:** §6.3 (LMNA:964-967) chooses plain SGD as "closer to single-pass
  human learning".
- **Problem:** Transformers train poorly with plain SGD compared with
  Adam-type optimisers (e.g. Zhang et al., 2024). That worsens the
  data-efficiency problem LMNA already flags (§6.5). The justification is
  speculative.
- **Fix:** Compare plain SGD with Adam-type and low-memory optimisers in C0.
  Report the optimiser's memory honestly: Adam adds two values per parameter.

#### N28. C has no grounding, so several tests become proxies — *Medium*

- **Where:** §6.4 (LMNA:983-995).
- **Problem:** C agents learn from text alone, so words have no referents. To
  "know" a word means to predict it in context. CDI comprehension, picture
  naming (V14) and translation equivalence have no direct analogue. Links
  between the languages come only from words co-occurring in bilingual text.
- **Fix:** Add a concept channel (concept tokens, or fixed semantic vectors
  attached to content words in the training stream), or state that V14 and CDI
  comprehension are only proxies for C.

#### N29. B's "emergent" age effects partly rest on an assumed age schedule — *Medium*

- **Where:** B0's acceptance criterion requires the attrition profile to match
  V4/V5 "**without** age-specific memory parameters" (LMNA:877). Yet the
  learning rate `η(age)` and neighbourhood width `σ(age)` shrink through
  childhood by design (LMNA:821-822).
- **Problem:** Those schedules *are* age curves. B's genuine advantage is
  different: a single schedule must fit both age-of-acquisition effects and the
  attrition profile, which is a real joint test.
- **Fix:** Reword the criterion and the summary table accordingly.

#### N30. B's attrition depends on map size, and its learning depends on order — *Medium*

- **Where:** §5.4 and the risk table in §5.9 (LMNA:908-916).
- **Problem:**
  - If the map is large enough, new `L2` input never has to take over `L1`
    territory, so forgetting by interference never happens. The V4/V5 contrast
    then hinges on map size, an arbitrary setting.
  - Online self-organising-map learning depends on the order of inputs, which
    conflicts with R5. Batch variants of the algorithm avoid this.
- **Fix:** Add both to the risk table, and test the sensitivity to map size.

### 3.7 Methodology

#### N31. There is no trivial baseline learner — *Medium*

- **Problem:** The plan compares A, B and C with each other and with legacy,
  but not with a simple statistical model.
- **Fix:** Add a plug-in that predicts proficiency in each language as a fitted
  function of cumulative and current exposure (curves like Thordardottir's and
  Unsworth's), plus an attrition term. If A does not beat it on held-out
  targets, its extra structure is not justified. It also gives a fast fallback
  for population runs.

#### N32. A has no cheap route for large ensembles — *Low*

- **Problem:** C has an emulator route. A relies on raw speed alone.
- **Fix:** A version of A with traces per *frequency band* of lemmas, with
  fixed meaning per band, could serve as an emulator for population
  calibration, validated against full A.

### 3.8 Minor slips

| Where | Issue |
|---|---|
| LMNA:179 | "One compressed token stands for a band of about 20 real words" holds only for adults. At birth a band is about 1.7 words (MMA M8). |
| LMNA:133 | `agent.py:846` is in `study_vocab`, not `pick_vocab`. |
| LMNA:207-209, LMNA:1275 | The 15 % spent in `update_memory_retention` comes mostly from recomputing `R` and `pct` over all 500 slots after *every* learning update (`agent.py:723`, 874k calls in 6 steps). The once-per-step decay sweep (`schedule.py:46`) is a small part. This matters for speeding up legacy. |

---

## 4. What each approach's risk table should add

| Approach | Risks to add | Findings |
|---|---|---|
| **A** | time unit and session scaling of the memory rule; no native plateau; compact-state error; closed lexicon with no variants; one-to-one filter; lexicon too small for adult tests; no pseudoword model; parameter count; dependence on cohort history | N1–N3, N8, N9, N11, N12, N16, N20 |
| **B** | assumed plasticity schedule; attrition depends on map size; order sensitivity; adult meanings from text-trained embeddings | N29, N30 |
| **C** | memory-bound cost; replay not costed; not enough child-directed text; plain SGD; no grounding | N25–N28 |

## 5. Suggested order for revising the documents

1. **Specify A's memory rule** (N1, N2, N4, X1): time unit, session scaling,
   candidates for D1, and V3 as a gating test. This changes the core of A0.
2. **Decide the lexicon** (N8, N9, N11, N12): size, many-to-many concepts,
   whether to include the variant layer, and the pseudoword response model.
   This changes the shared foundations.
3. **Fix the calibration design** (N13–N16): one target list, a clean
   fit/hold-out split, history matching followed by simulation-based
   inference, and an honest parameter count.
4. **Reconcile the two documents** (X1–X8): a note in MMA's status block,
   `legacy` vs `legacy-corrected`, the threshold mapping, and the age
   assumption.
5. **Extend the exposure model** (N17–N20): speaker type, burstiness, cohort
   histories.
6. **Correct the resource estimates** (N21–N25).
7. **Reword the claims for B and C** (N26–N30); add the baseline and the
   emulator (N31, N32).
8. **Fix the minor slips** (D4–D11, §3.8).

---

## Appendix A — ACT-R check: time unit and long-term forgetting

**Rule.** ACT-R base-level activation with Pavlik & Anderson's
activation-dependent decay, computed over the full history (no compact state):

```
m(T) = ln( Σ_j (T − t_j)^(−d_j) ),    d_j = c · exp(m(t_j)) + α,
c = 0.217, α = 0.177 (values commonly quoted from Pavlik & Anderson, 2005;
                      to be checked against the paper)
P(recall) = 1 / (1 + exp(−(m − τ)/s)),  τ = −0.7, s = 0.25 (illustrative)
```

**Scenarios.** A frequent word used 3 times a day (at 0.3, 0.5 and 0.7 of each
day) for 20 years; a school word used twice a week for 3 years. Use then stops.

| Unit | Scenario | Final decay | +1 day | +1 y | +5 y | +10 y | +25 y | +40 y |
|---|---|---|---|---|---|---|---|---|
| days | frequent word | **∞** | 0.97 | **0.04** | 0.03 | 0.03 | 0.02 | 0.02 |
| days | school word | 0.90 | 1.00 | 0.98 | 0.82 | 0.59 | 0.24 | 0.14 |
| seconds | frequent word | 0.53 | 0.99 | 0.98 | 0.96 | 0.94 | 0.85 | 0.76 |
| seconds | school word | 0.38 | 0.93 | 0.83 | **0.61** | 0.45 | **0.23** | 0.15 |

Values are recall probabilities. With time in days, gaps within a day are below
1 and the decay of the frequent word runs away to infinity (N1). With time in
seconds the rule is stable, but recall keeps falling on a log-time scale, with
no plateau (N2).

**Script** (run with `uv run python actr_check.py`):

```python
import numpy as np
C, ALPHA = 0.217, 0.177
TAU, S = -0.7, 0.25
YEAR = 365.0

def decays(times):
    d = np.empty_like(times)
    for j in range(len(times)):
        if j == 0:
            d[j] = ALPHA
        else:
            with np.errstate(over='ignore'):
                m = np.log(np.sum((times[j] - times[:j]) ** (-d[:j])))
            d[j] = C * np.exp(m) + ALPHA
    return d

def activation(times, d, T):
    with np.errstate(over='ignore'):
        return np.log(np.sum((T - times) ** (-d)))

scenarios = {
    'frequent word, 3 uses/day for 20 y': (np.arange(int(20 * YEAR))[:, None] + [0.3, 0.5, 0.7]).ravel(),
    'school word, 2 uses/week for 3 y':  np.arange(0, 3 * YEAR, 3.5),
}
delays = [('+1 d', 1), ('+1 y', YEAR), ('+5 y', 5 * YEAR), ('+10 y', 10 * YEAR),
          ('+25 y', 25 * YEAR), ('+40 y', 40 * YEAR)]
for unit_name, unit in [('days', 1.0), ('seconds', 86400.0)]:
    for name, enc_days in scenarios.items():
        t = enc_days * unit
        d = decays(t)
        row = []
        for lab, delay in delays:
            m = activation(t, d, (enc_days[-1] + delay) * unit)
            row.append(f"{lab}: P={1 / (1 + np.exp(-(m - TAU) / S)):.2f}")
        print(f"[{unit_name}] {name}: final d={d[-1]:.2f} | " + " | ".join(row))
```

## Appendix B — Cost of approach C when memory bandwidth is the limit

**Assumptions (as in LMNA's Appendix A):**
- `N = 1.2 × 10⁷` parameters, stored in bf16, so 24 MB of weights per agent;
- `3.65 × 10⁸` agent-days, with 10,000 tokens generated and 25,000 tokens
  trained on per agent-day;
- assumed throughput `2.5 × 10¹⁴` FLOP/s;
- H100 memory bandwidth of about `3.35 × 10¹²` bytes/s.

**Generation, one utterance at a time per agent.**
- Each new token reads all the agent's weights: `2N` FLOP for `2N` bytes, so
  about 1 FLOP per byte.
- Total weight traffic: `3.65 × 10¹²` tokens × `2.4 × 10⁷` bytes ≈
  `8.8 × 10¹⁹` bytes. At `3.35 × 10¹²` bytes/s that is `2.6 × 10⁷` s, about
  **7,300 GPU-hours**.
- The compute-bound estimate is `0.9 × 10²⁰ / 2.5 × 10¹⁴` s, about 100
  GPU-hours.
- Batching `b` utterances of the same agent divides the traffic by about `b`.

**Training, one plain-SGD step per utterance of `L` tokens.**
- The step costs `6NL` FLOP. Reading the weights for the forward and backward
  passes and writing them back moves at least `3 × 2N` bytes, so the
  intensity is about `L` FLOP per byte.
- For `L = 20`: about `6.7 × 10¹³` FLOP/s, 3.7× below the assumed throughput.
  `6.6 × 10²⁰` FLOP then takes about **2,700 GPU-hours**, instead of about 730.

**Total.** About 10,000 GPU-hours in the worst case, against 830 estimated,
before replay. Batching each agent's work over a whole day brings the cost back
near the estimate, with the trade-offs described in N25.

## Appendix C — Numbers re-checked

**Found correct** (by direct evaluation of the formulas, or from the code):

- **MMA §1.1:** `R` = 0.9, 0.81, 0.59, 0.35 and 0.1 at `t` = `S`, 2`S`, 5`S`,
  10`S` and 22`S`; half-life 6.6`S`.
- **MMA §1.2:** the table of the gain function `g`.
- **MMA §1.4:**
  - `S` = 0.83, then `R` = 0.88 and ≈ 0.32 after 100 days;
  - `S` = 14.6, with `R` = 0.77 one year later;
  - `S` = 38.5.
- **MMA M1:** SuperMemo's stability increase at `R` = 90 % is 2.2–2.7 for `S`
  of 1–100 days.
- **MMA M2 table:** 142.5, 9.4, 14.6 and 0.96; `R` one year later 0.97 and
  0.77.
- **MMA M3:** +12.9 vs +0.32 steps (40×); 0.16 at `R` = 0.95.
- **MMA M4/M11:** 74 % and 15 % of slots pass the similarity tests.
- **MMA M5:** exponential vs power law for `S = 50`: 0.47 and 0.05 vs 0.61 and
  0.36.
- **MMA M8:** the 6.5× and 7× ratios.
- **MMA M12:** `S` = 142.5, 24.9 and 9.4 for the three ways of splitting the
  calls.
- **MMA fix A2:** all three splits give 9.4.
- **MMA Option B:** `S` ≈ 16 for one use per step.
- **The two-pass update** described in MMA matches `agent.py:698-716`: the
  second pass uses the updated `S` and the old `R`.
- **LMNA Appendix A:**
  - A: 1.1 × 10¹² updates per run;
  - B0: 2.3 MB per agent;
  - C: training 6.6 × 10²⁰ FLOP, generation 0.9 × 10²⁰ FLOP, ≈ 830 GPU-hours
    in total;
  - C0: 9.5 × 10¹⁷ FLOP, ≈ 9 h.
- **LMNA Appendix B:** 0.21 s per agent per simulated year, so about 60 h for
  10,000 agents over 100 years.

**Found wrong or inconsistent:** see D11, N22 and N23.

## Appendix D — References not cited in the reviewed documents

- Adelman, J. S., Brown, G. D. A., & Quesada, J. F. (2006). Contextual diversity, not word frequency, determines word-naming and lexical decision times. *Psychological Science*, 17(9), 814–823.
- Altmann, E. G., Pierrehumbert, J. B., & Motter, A. E. (2009). Beyond word frequency: bursts, lulls, and scaling in the temporal distributions of words. *PLoS ONE*, 4(11), e7678.
- Andrianakis, I., Vernon, I. R., McCreesh, N., et al. (2015). Bayesian history matching of complex infectious disease models using emulation: a tutorial and a case study on HIV in Uganda. *PLoS Computational Biology*, 11(1), e1003968.
- Baxter, G. J., Blythe, R. A., Croft, W., & McKane, A. J. (2006). Utterance selection model of language change. *Physical Review E*, 73, 046118.
- Blythe, R. A., & Croft, W. (2012). S-curves and the mechanisms of propagation in language change. *Language*, 88(2), 269–304.
- Bowers, J. S., Mattys, S. L., & Gage, S. H. (2009). Preserved implicit knowledge of a forgotten childhood language. *Psychological Science*, 20(9), 1064–1069.
- Choi, J., Broersma, M., & Cutler, A. (2017). Early phonology revealed by international adoptees' birth language retention. *PNAS*, 114(28), 7307–7312.
- Church, K. W. (2000). Empirical estimates of adaptation: the chance of two Noriegas is closer to p/2 than p². *Proc. COLING 2000*.
- Gonzalez-Agirre, A., Laparra, E., & Rigau, G. (2012). Multilingual Central Repository version 3.0.
- Kornell, N., Hays, M. J., & Bjork, R. A. (2009). Unsuccessful retrieval attempts enhance subsequent learning. *JEP: Learning, Memory, and Cognition*, 35(4), 989–998.
- Krashen, S. D., Long, M. A., & Scarcella, R. C. (1979). Age, rate and eventual attainment in second language acquisition. *TESOL Quarterly*, 13(4), 573–582.
- Pagel, M., Atkinson, Q. D., & Meade, A. (2007). Frequency of word-use predicts rates of lexical evolution throughout Indo-European history. *Nature*, 449, 717–720.
- Peterson, R. R., & Savoy, P. (1998). Lexical selection and phonological encoding during language production: evidence for cascaded processing. *JEP: Learning, Memory, and Cognition*, 24(3), 539–557.
- Pierce, L. J., Klein, D., Chen, J.-K., Delcenserie, A., & Genesee, F. (2014). Mapping the unconscious maintenance of a lost first language. *PNAS*, 111(48), 17314–17319.
- Place, S., & Hoff, E. (2011). Properties of dual language exposure that influence 2-year-olds' bilingual proficiency. *Child Development*, 82(6), 1834–1849.
- Poplack, S., Sankoff, D., & Miller, C. (1988). The social correlates and linguistic processes of lexical borrowing and assimilation. *Linguistics*, 26(1), 47–104.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249–255.
- Slamecka, N. J., & Graf, P. (1978). The generation effect: delineation of a phenomenon. *JEP: Human Learning and Memory*, 4(6), 592–604.
- Vernon, I., Goldstein, M., & Bower, R. G. (2010). Galaxy formation: a Bayesian uncertainty analysis. *Bayesian Analysis*, 5(4), 619–669.
- Walsh, M. M., Gluck, K. A., Gunzelmann, G., Jastrzembski, T., Krusmark, M., Myung, J. I., Pitt, M. A., & Zhou, R. (2018). Mechanisms underlying the spacing effect in learning: a comparison of three computational models. *JEP: General*, 147(9), 1325–1348.
- Weisleder, A., & Fernald, A. (2013). Talking to children matters: early language experience strengthens processing and builds vocabulary. *Psychological Science*, 24(11), 2143–2152.
- Wozniak, P. A., Gorzelańczyk, E. J., & Murakowski, J. A. (1995). Two components of long-term memory. *Acta Neurobiologiae Experimentalis*, 55.
- Zhang, Y., Chen, C., Ding, T., Li, Z., Sun, R., & Luo, Z.-Q. (2024). Why transformers need Adam: a Hessian perspective. *NeurIPS 2024*.
