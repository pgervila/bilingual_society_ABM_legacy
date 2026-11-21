# Five essential language features, and whether they need embeddings

> **Status (Nov 2026):** analysis only. No code was changed. This document
> builds on [`memory_model_analysis.md`](../memory_model_analysis.md) (**MMA**,
> findings M1–M13), on
> [`learning_model_new_approaches.md`](../learning_model_new_approaches.md)
> (**LMNA**: shortcomings S1–S17, requirements R1–R8, validation targets
> V1–V18, approaches A/B/C) and on
> [`learning_model_docs_review.md`](../learning_model_docs_review.md) (review
> findings X, D, N). It proposes new validation targets **V19–V23** (§9.2). The
> general question of embeddings and vector databases is answered in
> [`embeddings_and_vector_databases.md`](embeddings_and_vector_databases.md).

---

## 0. Summary

`bilangsim` asks when minority languages survive in bilingual societies. To
answer that realistically, the learning core must represent five features of
language. **Every one of them can be captured without embeddings** at the level
the research question needs. Only the *emergence* of grammar from raw input
goes beyond that, and it needs a neural learner (approach C), not imported
embeddings.

| # | Feature | Why it is essential | Embedding-free mechanism | Main data | Without embeddings? |
|---|---|---|---|---|---|
| **F1** | Frequency-structured, growing lexicon | Every proficiency measure counts which words are known. Comprehension depends on token coverage. | Real lemmas with per-register frequency and contextual diversity. Bursty sampling. Vocabulary growth emerges from exposure and memory. | SUBTLEX-CAT/ESP, CHILDES, prevalence and AoA norms, Wordbank | **fully** |
| **F2** | Form overlap between the languages: cognates, false friends, distance between the languages | Catalan–Spanish closeness makes receptive bilingualism cheap and shapes lexical access. Results must generalise to distant pairs. | Edit distance on spelling and IPA between translation pairs; false-friend list; the distribution of distances is the language distance | Apertium spa–cat, espeak-ng | **fully, and better than with embeddings** |
| **F3** | Meaning, domains and context | Shift proceeds domain by domain. Policies act on school and work vocabulary. Meaning inference drives incidental learning. | Per-setting word frequencies; domain labels where no corpus exists; co-occurrence (PPMI) table for support from context | register corpora, MCR wordnets (Catalan and Spanish), CHILDES | **mostly**: embeddings optional, offline, gated |
| **F4** | Bilingual lexical access: comprehension vs production, competition, language control, code-switching | The social layer runs on who understands and who speaks. Receptive bilinguals keep the minority language usable. | Localist shared-concept architecture (BIA+, Multilink, RHM): competing lemmas, control bias, language-level inhibition, two thresholds | test data (V8, V12, V14), code-switching corpora | **fully** at word level |
| **F5** | Grammar and contact-induced change: loans, calques, structural convergence | A language can keep its speakers and still converge on the majority language. "Speaks L" includes grammar. | Inventory of lexical variants and contact-sensitive grammatical variables. Each variant has a memory trace and competes in production. Age-limited plasticity. | contact studies (e.g. Catalan clitics *en*/*hi*), spoken corpora | **partly**: measurable contact phenomena yes; emergent grammar no |

**Main conclusions.**

1. **The biggest realism gaps are not about semantics.** They are F3's domains
   and F5's contact change. Data closes them: register corpora, variant
   inventories, measured variant rates. Embeddings do not.
2. **The leading computational models of bilingual lexical access are
   localist.** Multilink, BIA+ and RHM, which A builds on, use no embeddings.
   Multilink computes form overlap with a length-normalised Levenshtein distance
   (Dijkstra et al., 2019).
3. **Embeddings earn a place only in F3**, as an offline source of graded
   semantic similarity, behind the gate in the companion document (§7 there).
4. **Emergent grammar is the one real limit** of an embedding-free design. It
   needs approach C. C's representations are learned, not imported, so the
   question there is whether to use neural agents, not whether to use
   embeddings.

---

## 1. How the five were chosen

A feature is called *essential* if it meets all three criteria:

1. **It changes the outcomes under study.** It changes what agents know of each
   language in a way that reaches social decisions (MAXIMIN, transmission,
   friendship, jobs) or the state of the minority language itself.
2. **It can be tested.** Data exists or can be collected (LMNA §3.8) that the
   model's version of the feature can be checked against.
3. **It is missing or distorted today** (S1–S17, M1–M13).

Realistic *learning* (spacing, forgetting, age, exposure) is not one of the
five. It applies to all of them, and §7 covers it.

Each feature section follows the same plan:
- why the feature is essential;
- what realism requires;
- what the current model does;
- a mechanism without embeddings;
- what embeddings would add;
- the verdict.

---

## 2. F1 — A frequency-structured, growing lexicon

**Why it is essential.**
- Every measure of proficiency counts which words an agent knows: CDI
  checklists, vocabulary size, LexTALE, coverage, census self-reports.
- Word frequencies span several orders of magnitude. About 2,000–3,000 word
  families cover about 95% of spoken English (Adolphs & Schmitt, 2003).
- So the frequency structure fixes what "understands" means. The size of the
  lexicon decides whether adult tests can be simulated at all (review N11).

**What realism requires.**
- V6: frequent words are known by more people and recognised faster.
- V7: comprehension needs about 95–98% coverage.
- V9: vocabulary grows plausibly, from CDI ages to about 42,000 lemmas at
  20 years.
- V11: the share of exposure to a language predicts vocabulary in it.
- Contextual diversity (the number of contexts a word occurs in) predicts
  recognition times better than raw frequency (Adelman, Brown & Quesada, 2006).
- Word occurrences are bursty: a word that has just appeared is likely to recur
  (Church & Gale, 1995; Altmann, Pierrehumbert & Motter, 2009). Burstiness sets
  how massed exposure is, which is exactly what a spacing-sensitive memory rule
  responds to (review N19).

**What the current model does.**
- 500 slots whose meaning changes with age (M8, S1).
- A knowledge measure that counts word types (M7).
- Independent draws from one age-specific distribution: no burstiness and no
  contextual diversity.

**Mechanism without embeddings.**
- **Inventory.** Real lemmas, enough of them for the tests: at least
  20,000–30,000 per language for adult vocabulary tests (review N11). Add a
  separate class for function words, which counts toward coverage (review N9).
- **Frequencies per register.** Adult conversation from SUBTLEX-CAT and
  SUBTLEX-ESP. Child-directed speech from CHILDES. Contextual diversity is
  stored with each lemma: SUBTLEX-CAT reports it, and for Spanish it can be
  computed from the subtitle corpus if SUBTLEX-ESP lacks it.
- **Bursty sampling.**
  - Each conversation first draws a topic (F3), then words from that topic.
  - Alternatively, a per-conversation count is drawn from a Poisson mixture or
    negative binomial whose dispersion is fitted to the corpus.
  - Either way, words recur within conversations as they do in real speech.
- **Growth.** Growth comes from exposure and memory. There is no ceiling curve:
  a child knows few words because it has met few, not because a
  `vocab_ceiling_curve` caps the sampling.
- **Measures.** Token coverage, known-lemma counts (receptive and productive)
  and test scores (LMNA §3.4).

**What embeddings would add.** Nothing. Frequency, contextual diversity,
burstiness, age-of-acquisition and prevalence are all counts or norms.

**Verdict: fully capturable without embeddings.**

---

## 3. F2 — Form overlap: cognates, false friends, distance between the languages

**Why it is essential.**
- Catalan and Spanish are close. Many translation pairs are cognates (`aigua` /
  `agua`, `nit` / `noche`, `porta` / `puerta`).
- That closeness is why a Spanish monolingual can understand a good deal of
  Catalan with little exposure. It makes receptive bilingualism cheap, which
  matters for MAXIMIN and so for which language conversations use.
- Cognates are named faster, especially in the non-dominant language (Costa,
  Caramazza & Sebastián-Gallés, 2000). They are learned faster and forgotten
  more slowly (de Groot & Keijzer, 2000).
- False friends work against comprehension: Catalan `cama` means "leg", Spanish
  `cama` means "bed".
- To say whether a Catalan–Spanish conclusion holds for Basque–Spanish, the
  model must represent the distance between the two languages (LMNA §2.3b).

**What realism requires.**
- V14: the cognate naming advantage.
- Proposed **V19**: comprehension of an unlearned closely related language
  rises with phonetic and lexical overlap. For the mainland Scandinavian
  languages, phonetic distance (Levenshtein on transcriptions) predicted
  intelligibility better than lexical distance (Gooskens, 2007). A
  Catalan–Spanish listening test for monolinguals is the matching instrument,
  and it can be collected.

**What the current model does.** "Similarity" is a random binomial edit distance
per slot: 15% of slots are "similar" for the pure stores and 74% for the mixed
stores (M11, L3).

**Mechanism without embeddings.**
- **Translation pairs** come from the concept table (Apertium spa–cat,
  many-to-many, review N9).
- **Cognate strength.** For each pair, `σ ∈ [0, 1]` from the length-normalised
  Levenshtein distance between spellings (rapidfuzz) and between IPA
  transcriptions (espeak-ng). An articulatory-feature-weighted distance
  (PanPhon; Mortensen et al., 2016) is a later refinement, in the tradition of
  dialect measurement (Heeringa, 2004).
- **False friends.** Pairs of *forms* across the two lemma lists with high
  similarity but different concepts. They are found by comparing forms, not
  translation pairs, and they can lower recognition or cause misunderstanding.
- **Distance between the languages.** No parameter is needed. It is the
  frequency-weighted distribution of `σ` over the concept table. For another
  language pair, replace the lexicon and the distance follows.
- **Use in the learner.** Start with a single cognate path: the cognate route in
  comprehension (LMNA §4.5). Add item difficulty or support for meaning
  inference only when a target requires them, since three paths serving one
  target cannot be told apart (review N6).

This is how the reference model works. Multilink is a localist-connectionist
model of bilingual word recognition and translation. Its orthographic
activation comes from Levenshtein distance normalised by word length (Dijkstra
et al., 2019).

**What embeddings would add.** Nothing essential. Semantic embeddings encode
meaning, not form, and fastText's character n-grams blur the two. Aligned
embeddings can suggest translation candidates when the concept table is being
cleaned, which is only a quality check.

**Verdict: fully capturable without embeddings, and explicit form measures are
the better tool.**

---

## 4. F3 — Meaning, domains and context

**Why it is essential.**
- Languages are lost and kept domain by domain: home, school, work,
  administration, media (Fishman, 1991). The classic shift pattern has the
  minority language retreat to the home.
- School and workplace language policies act on *those domains'* vocabulary.
- Bilingual lexical knowledge is distributed: some words are known in one
  language only (Oller, Pearson & Cobo-Lewis, 2007).
- Bialystok et al. (2010) tested 1,738 children aged 3–10, schooled in English
  with various home languages. Bilinguals' receptive-vocabulary deficit was
  largely confined to words of the home context. On school words they matched
  monolinguals.
- Inferring the meaning of an unknown word from its context drives incidental
  learning (V15).

**What realism requires.**
- Proposed **V20**: the home-vs-school vocabulary profile (Bialystok et al.,
  2010).
- V15: incidental learning needs repeated encounters.
- Cross-situational learning: word meanings are inferred from regularities
  across situations (Yu & Smith, 2007).

**What the current model does.** Setting never enters the choice of words. A
teacher, parents at dinner and colleagues at work all draw from the same
age-specific distribution (S2). Domain profiles cannot emerge.

**Mechanism without embeddings.**

1. **Word frequencies by setting.** Estimate `P(c | setting, age)` directly
   wherever a register corpus exists:
   - CHILDES for home with young children;
   - subtitles by genre for adult conversation and media;
   - textbooks or classroom transcripts for school, by grade.

   Where a setting has no corpus (work sectors, administration), use the domain
   labels of concepts. The Multilingual Central Repository (MCR) has Catalan and
   Spanish wordnets linked through the WordNet 3.0 Inter-Lingual Index, and it
   integrates domain labels such as WordNet Domains. Then
   `P(c | setting) ∝ P(c) · (1 + λ_setting · [c ∈ domain(setting)])`, with one
   parameter per type of setting. Check the result against held-out samples
   from the register, where any exist.
2. **Support from context for meaning inference.**
   - The simplest version, with the fewest parameters: `p_inf` depends on
     coverage only (LMNA §4.5).
   - If V15 or V20 needs more, add a sparse **PPMI co-occurrence table** from
     the corpora, keeping the top-k associates per concept. Let `p_inf` rise
     with the unknown concept's mean PPMI with the recognised context.
   - A probabilistic cross-situational model with symbolic semantic features
     learns word meanings from child-directed utterances (Fazly, Alishahi &
     Stevenson, 2010). It is the reference if meaning acquisition itself
     becomes a target.
3. **Polysemy.** Treat the dominant sense first. Move to sense-level concepts
   (MCR synsets) only for targets that need it.
4. **Burstiness for free.** A conversation draws a topic and then its words, so
   F1's burstiness comes from the same mechanism.

**What embeddings would add.**
- Graded semantic similarity at full-lexicon scale, for semantic competitors
  and priming (Mandera, Keuleers & Brysbaert, 2017).
- Domain word lists for settings that have neither a corpus nor domain labels,
  built from seed words and checked afterwards.
- A richer signal of support from context.

They also bring known problems:
- association is confused with similarity (Hill, Reichart & Korhonen, 2015);
- the meanings are adult and written;
- neighbours are unstable, more so for the smaller Catalan corpora (Antoniak &
  Mimno, 2018).

The companion document proposes a three-way experiment to settle the question
empirically (§8 there): no semantics, explicit semantics, embeddings.

**Verdict: mostly capturable without embeddings.** The part the research
question needs, domain-specific vocabulary profiles, is better done with
explicit per-setting counts, which can be checked against corpora. Embeddings
are an optional offline aid for coverage gaps.

---

## 5. F4 — Bilingual lexical access: comprehension vs production, competition, control, code-switching

**Why it is essential.**
- The social layer runs on who understands and who speaks each language
  (MAXIMIN, LMNA §3.1).
- *Receptive* bilinguals understand a language but rarely speak it. They keep
  the minority language usable in mixed conversations. Censuses and the EULP
  measure understanding and speaking separately.
- Dominance shifts and code-switching are observable outcomes of contact.

**What realism requires.**
- V8: receptive knowledge exceeds productive (the heritage profile).
- V12: receptive vocabulary reaches monolingual levels at lower exposure than
  expressive vocabulary.
- V14: cognate effects.
- Reduced L1 fluency during L2 immersion (Linck, Kroll & Sunderman, 2009).
  - Levy et al. (2007) attributed L1 loss partly to retrieval-induced
    forgetting: naming in L2 inhibited the L1 names.
  - Runnqvist & Costa (2012) tested that account. Their replication found the
    opposite pattern.
  - So inhibition is a hypothesis to test, not a given.
- Code-switching rates in corpora, which are higher for balanced bilinguals.
- Bilinguals' language mode depends on the listener (Grosjean, 2001).

**What the current model does.**
- One trace serves comprehension and production (M9).
- The `L12`/`L21` cascade reinforces the wrong trace and inflates `L1` (M4).
- The two languages do not compete (M10).

**Mechanism without embeddings.** A localist architecture, as in BIA+, Multilink
and the Revised Hierarchical Model:
- **Structure.** Shared concept nodes and language-tagged lemmas. Each lemma has
  one trace with a comprehension threshold and a stricter production threshold;
  a separate production trace is a later option.
- **Production.** The target lemma competes with its translation equivalent
  under a control bias toward the intended language (Green, 1998). The bias is
  larger with monolingual listeners.
- **Inhibition.** A per-language inhibition state, raised by use of the other
  language, lowers access to the inhibited language. It is needed to produce
  reduced fluency, not just more switching (review N5). It is calibrated on
  Linck et al. and tested against the mixed retrieval-induced-forgetting
  evidence above.
- **Reinforcement and input.** The lemma actually used is the one reinforced,
  and listeners hear what was said. This fixes M4: code-switched speech teaches
  the other language's words.
- **Where in the sentence switches happen.** That needs syntax (F5), not
  embeddings. Two options:
  - a rule such as Poplack's (1980) equivalence constraint;
  - a connectionist sentence-production model. The Bilingual Dual-path model,
    a recurrent network trained on sentences paired with a symbolic message,
    produced code-switches without ever seeing code-switched input, and it
    switched more when simulating balanced bilinguals (Tsoukala et al., 2021).

**What embeddings would add.** Only semantic competitors other than translation
equivalents, e.g. saying "cat" for "dog". This is a second-order effect for the
research question, and WordNet sister terms cover it if needed.

**Verdict: fully capturable without embeddings at word level.** The patterns of
switching within sentences depend on grammar (F5).

---

## 6. F5 — Grammar and contact-induced change

**Why it is essential.**
- A minority language can keep its speakers and still converge on the majority
  language:
  - loanwords replace native words;
  - calques enter the language;
  - grammatical features that have no counterpart in the majority language
    weaken.
- For sustainability, the *state of the language* matters as well as the number
  of speakers.
- Catalan has a well-documented case. Its non-personal clitics *en* and *hi*
  have no grammatical equivalent in Spanish.
  - **Adults** (Perpiñán, 2017; 57 participants born and raised in Catalonia):
    Spanish-dominant early bilinguals accepted ungrammatical omission and
    doubling of these clitics. The author argues that age of onset does not
    explain the differences between groups on its own: the quantity and quality
    of input matter. That is exactly what an exposure-driven learner can
    represent.
  - **Children** (Soto-Corominas, 2018): bilingual and Spanish-dominant
    children acquire Catalan partitive *en* and neuter *ho* later than
    Catalan-dominant children, and often have not caught up by age 8.
- In the lexicon, *vale* or *bueno* used inside Catalan speech are either nonce
  code-switches or established loans. The two behave differently (Poplack,
  Sankoff & Miller, 1988; review N8).
- "Speaks L" in a census includes grammar. Grammar-learning ability stays high
  until about 17 and then declines (Hartshorne, Tenenbaum & Pinker, 2018; V18).

**What realism requires.**
- V18: the age profile of grammar learning.
- Proposed **V21**: contact-variant rates by language dominance and age
  (Perpiñán, 2017; Soto-Corominas, 2018).
- Proposed **V22**: a new variant spreads through a community along an S-curve
  (Blythe & Croft, 2012).
- Proposed **V23**: established loans are integrated and used even by
  monolinguals; nonce borrowings are not (Poplack et al., 1988).

**What the current model does.** Nothing. Languages are bags of word-frequency
tokens (legacy analysis, weakness 9).

**Mechanism without embeddings.** Model each contact phenomenon as a set of
**variants competing for one function**:

- **Inventory.**
  - *Lexical variants* per concept: native, dialectal, normative, loan
    (review N8).
  - *Grammatical variables*: start with 5–10 contact-sensitive ones whose rates
    can be measured. Examples:
    - partitive *en* expressed vs omitted;
    - *hi* expressed vs omitted;
    - an obligation construction with *haver de* vs the commonly cited calque
      *tenir que*.
- **Memory.** Each (agent, variant) pair has a trace governed by the same memory
  rule as lemmas, treating constructions as stored pairings of form and
  function (Goldberg, 2006). This reuses the lexical machinery and adds few
  parameters.
- **Production.** The agent chooses among a variable's variants by relative
  activation, plus three biases:
  - a normative bias from school and media;
  - accommodation to the listener;
  - for loans, a learned weight for which language the variant belongs to
    (review N8).
- **Age.** Plasticity for grammatical variants declines after late adolescence
  (V18).
- **Alternative learner.** Yang's variational learner (linear reward–penalty)
  links acquisition to change across generations (Yang, 2000). It is worth
  comparing on V21 and V22.
- **Community.** In aggregate, this is the utterance-selection model of language
  change (Baxter, Blythe, Croft & McKane, 2006), which yields the S-curves of
  V22.
- **Data.**
  - Variant rates by age and dominance in spoken corpora.
  - Acceptability-judgement and elicitation studies, such as Perpiñán's and
    Soto-Corominas'.
  - A short grammar module added to the planned online vocabulary study
    (LMNA §3.8): a few acceptability items per variable.

**What this cannot do.** It cannot produce new constructions, handle forms it was
never given, or capture interactions across a whole grammar in real sentences.
That requires a learner that induces grammar from input, such as approach C or
a dual-path-style network. Such a learner learns its own distributed
representations. Embeddings are then part of the learner, not an imported
resource, and C's costs and validation problems apply (LMNA §6; review
N25–N28).

**Verdict: partly capturable without embeddings.** The measurable contact
phenomena, which are what sustainability studies need (which variants, how
fast, carried by whom), are captured by design. Emergent grammar is not. This
is the one place where the embedding-free design gives something up. What it
gives up is emergence, not reliability.

---

## 7. Realistic learning across the five features

Every learning property the plan needs works per item or per agent. None needs
embeddings.

| Learning property | Where it lives | Without embeddings? | Notes |
|---|---|---|---|
| Spacing, power-law-like forgetting, lapses | the memory rule per trace | yes | Candidates: ACT-R with Pavlik–Anderson decay, the multiscale context model, FSRS, PPE, compared at D1 (review N4). The time unit must be fixed first (review N1). |
| Permastore plateau (V3) | an explicit consolidation mechanism | yes | A pure power law has no plateau (review N2). Use multiscale or two-component traces. |
| Speaking vs listening | encoding weights per event type | yes | Priors from the testing and generation effects (review N7). |
| Directed vs overheard input; native vs non-native speakers | the exposure model | yes | review N18 |
| Massing and burstiness | topic-conditioned sampling (F1, F3) | yes | review N19 |
| Age effects on vocabulary | fitted age curves (A) | yes | Emergent age effects need distributed maps (B, below). |
| Age effects on grammar | plasticity of grammatical variants (F5) | yes | V18 |
| Individual differences | per-agent traits, e.g. rate of forgetting | yes | Sense et al. (2016) |
| Meaning inference | F3: coverage, PPMI, cross-situational learning | yes | Fazly et al. (2010) |

**Emergent age effects are not a reason for embeddings either.** In approach B,
entrenchment and the child-vs-adult attrition contrast come from competition for
map territory under declining plasticity. They do not come from the embeddings.
B's semantic input could be hand-built semantic features instead, such as
feature norms (McRae et al., 2005) or features derived from WordNet. Running B0
with both kinds of input would show whether B's conclusions depend on the
embeddings (§10).

---

## 8. What was left out, and why

| Feature | Why it is not among the five | Where it belongs |
|---|---|---|
| Phonology and accent | Marks identity and affects social acceptance, but barely changes *whether* a language is used. Its effect on intelligibility is covered through IPA distance (F2). | later: accent as an agent trait that affects listeners' reactions |
| Literacy | Censuses measure reading and writing (S17), but literacy is a separate channel with its own exposure. It needs no embeddings: a reading register plus orthographic knowledge. | stage A2 (LMNA §4.10) |
| Pragmatics and norms of language choice | These decide which language is used, not what is known. | the social layer: MAXIMIN, stickiness within relationships, accommodation |
| Attitudes and identity | Probably the strongest drivers of shift, but social-psychological rather than linguistic | the social layer |

---

## 9. An embedding-free design that covers the five features

### 9.1 Layers

| Layer | State per agent | Shared data | Mechanism | Targets | Stage |
|---|---|---|---|---|---|
| **Lexicon (F1)** | one trace per (agent, lemma) | lemma list; frequency and contextual diversity per register | memory rule; bursty sampling | V6, V7, V9, V11 | A0 |
| **Form (F2)** | none (shared) | concept table; spelling and IPA distances; false-friend list | cognate route in comprehension | V14, V19 | A0 |
| **Domains (F3)** | none extra: profiles emerge in the lexicon state | P(c \| setting) from register corpora plus MCR domain labels; optional PPMI table | setting-conditioned, topic-first sampling; support from context | V15, V20 | A1 |
| **Access (F4)** | inhibition state per language; optional separate production trace | — | competition, control bias, inhibition, reinforcement of the lemma used | V8, V12, Linck et al., switching rates | A0 (competition), A1 (inhibition) |
| **Contact and grammar (F5)** | traces per (agent, variant) for lexical variants and grammatical variables; language weights for loans | variant inventory; variant rates | competition among variants with normative bias; age-limited plasticity | V18, V21–V23 | new stage, after A1 |

### 9.2 Proposed validation targets V19–V23

These extend LMNA §3.5.

| # | Finding | Source | Level |
|---|---|---|---|
| V19 | Comprehension of an unlearned closely related language rises with phonetic and lexical overlap; phonetic distance is the better predictor | Gooskens (2007), Scandinavian languages; a Catalan–Spanish listening test can be collected | single agent |
| V20 | Bilingual children's receptive deficit relative to monolinguals is concentrated in home vocabulary, not school vocabulary | Bialystok et al. (2010) | single agent |
| V21 | Contact variants by dominance: Spanish-dominant early bilinguals accept omission and doubling of *en*/*hi*; bilingual and Spanish-dominant children acquire *en* and *ho* later | Perpiñán (2017); Soto-Corominas (2018) | single agent |
| V22 | A new variant spreads through a community along an S-curve | Blythe & Croft (2012) | population |
| V23 | Established loans are integrated and used by monolinguals; nonce borrowings are not | Poplack, Sankoff & Miller (1988) | population (corpora) |

### 9.3 Cost

- **Lexicon.** At A4 scale (~60,000 lemmas over both languages) and 32–40 bytes
  per trace (review N22): about 1.9–2.4 MB per agent, or 19–24 GB at 10,000
  agents.
- **Variant layer.** About 1.2–2× that (review N8).
- **Grammatical variables and inhibition state.** Negligible: tens of numbers
  per agent.
- **Shared tables.** Concept table, distances, PPMI and domain labels: under
  100 MB.
- **Sparse storage.** Most agents know only part of the lexicon, so storing
  only touched traces would cut memory further.
- **Compute.** No GPU. Costs stay within the CPU tiers of LMNA §4.11.

---

## 10. Implications for the plan in LMNA

1. **State A1's semantics as explicit by default.** Per-setting frequencies,
   MCR domain labels and an optional PPMI table. Make embeddings a gated
   variant (companion document §7–8). This resolves review N10.
2. **Specify the form layer.** Distances on both spelling and IPA, a
   false-friend list, and a single cognate path at first (review N6). This
   replaces the random edit distances of the legacy model.
3. **Add a contact-and-grammar stage.** It contains the variant layer
   (review N8) and 5–10 grammatical variables, with targets V21–V23. Add a
   short acceptability module to the online study (LMNA §3.8) so the data
   exists when the stage starts.
4. **Run B0 with two kinds of semantic input**: aligned embeddings and
   hand-built features. If B's emergent effects depend on which one is used,
   that is a finding about B.
5. **Leave C as the only route to emergent grammar.** It needs no vector
   database. Its embeddings are learned inside the learner.
6. **Drop vector databases from consideration** (companion document §5).

---

## References

References already listed in LMNA Appendix C (Costa et al., 2000; de Groot &
Keijzer, 2000; Dijkstra & van Heuven, 2002; Fishman, 1991; Green, 1998;
Grosjean, 2001; Hartshorne et al., 2018; Linck et al., 2009; McClelland et al.,
1995; Sense et al., 2016; resources such as Apertium, CHILDES, SUBTLEX) and in
the review are not repeated.

- Adelman, J. S., Brown, G. D. A., & Quesada, J. F. (2006). Contextual diversity, not word frequency, determines word-naming and lexical decision times. *Psychological Science*, 17(9).
- Adolphs, S., & Schmitt, N. (2003). Lexical coverage of spoken discourse. *Applied Linguistics*, 24(4), 425–438.
- Altmann, E. G., Pierrehumbert, J. B., & Motter, A. E. (2009). Beyond word frequency: bursts, lulls, and scaling in the temporal distributions of words. *PLoS ONE*, 4(11).
- Antoniak, M., & Mimno, D. (2018). Evaluating the stability of embedding-based word similarities. *TACL*, 6, 107–119.
- Baxter, G. J., Blythe, R. A., Croft, W., & McKane, A. J. (2006). Utterance selection model of language change. *Physical Review E*, 73, 046118.
- Bialystok, E., Luk, G., Peets, K. F., & Yang, S. (2010). Receptive vocabulary differences in monolingual and bilingual children. *Bilingualism: Language and Cognition*, 13(4), 525–531.
- Blythe, R. A., & Croft, W. (2012). S-curves and the mechanisms of propagation in language change. *Language*, 88(2).
- Church, K. W., & Gale, W. A. (1995). Poisson mixtures. *Natural Language Engineering*, 1(2).
- Dijkstra, T., Wahl, A., Buytenhuijs, F., van Halem, N., Al-jibouri, Z., de Korte, M., & Rekké, S. (2019). Multilink: a computational model for bilingual word recognition and word translation. *Bilingualism: Language and Cognition*, 22(4) (online 2018).
- Fazly, A., Alishahi, A., & Stevenson, S. (2010). A probabilistic computational model of cross-situational word learning. *Cognitive Science*, 34(6), 1017–1063.
- Goldberg, A. E. (2006). *Constructions at Work: The Nature of Generalization in Language*. Oxford University Press.
- Gonzalez-Agirre, A., Laparra, E., & Rigau, G. (2012). Multilingual Central Repository version 3.0. *LREC 2012*. (Catalan and Spanish wordnets linked through the WordNet 3.0 Inter-Lingual Index; non-English data under CC BY 3.0.)
- Gooskens, C. (2007). The contribution of linguistic factors to the intelligibility of closely related languages. *Journal of Multilingual and Multicultural Development*, 28(6), 445–467.
- Heeringa, W. (2004). *Measuring Dialect Pronunciation Differences Using Levenshtein Distance*. PhD thesis, University of Groningen.
- Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: evaluating semantic models with (genuine) similarity estimation. *Computational Linguistics*, 41(4).
- Levy, B. J., McVeigh, N. D., Marful, A., & Anderson, M. C. (2007). Inhibiting your native language: the role of retrieval-induced forgetting during second-language acquisition. *Psychological Science*, 18(1), 29–34.
- Mandera, P., Keuleers, E., & Brysbaert, M. (2017). Explaining human performance in psycholinguistic tasks with models of semantic similarity based on prediction and counting. *Journal of Memory and Language*, 92, 57–78.
- McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. *Behavior Research Methods*, 37(4).
- Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C., & Levin, L. (2016). PanPhon: a resource for mapping IPA segments to articulatory feature vectors. *COLING 2016*.
- Oller, D. K., Pearson, B. Z., & Cobo-Lewis, A. B. (2007). Profile effects in early bilingual language and literacy. *Applied Psycholinguistics*, 28(2).
- Perpiñán, S. (2017). Catalan-Spanish bilingualism continuum: the expression of non-personal Catalan clitics in the adult grammar of early bilinguals. *Linguistic Approaches to Bilingualism*, 7(5), 477–513.
- Poplack, S. (1980). Sometimes I'll start a sentence in Spanish y termino en español: toward a typology of code-switching. *Linguistics*, 18(7–8).
- Poplack, S., Sankoff, D., & Miller, C. (1988). The social correlates and linguistic processes of lexical borrowing and assimilation. *Linguistics*, 26(1).
- Runnqvist, E., & Costa, A. (2012). Is retrieval-induced forgetting behind the bilingual disadvantage in word production? *Bilingualism: Language and Cognition*, 15(2), 365 ff.
- Soto-Corominas, A. (2018). PhD dissertation on the acquisition of Catalan clitics by Catalan–Spanish bilingual children, Western University. <https://ir.lib.uwo.ca/etd/5441>
- Tsoukala, C., Broersma, M., van den Bosch, A., & Frank, S. L. (2021). Simulating code-switching using a neural network model of bilingual sentence production. *Computational Brain & Behavior*, 4, 87–100.
- Yang, C. D. (2000). Internal and external forces in language change. *Language Variation and Change*, 12(3), 231–250.
- Yu, C., & Smith, L. B. (2007). Rapid word learning under uncertainty via cross-situational statistics. *Psychological Science*, 18(5).
