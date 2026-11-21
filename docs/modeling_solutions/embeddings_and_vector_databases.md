# Are embeddings and vector databases essential for reliable modelling?

> **Status (Nov 2026):** analysis only. No code was changed. This document
> answers one question about the plan in
> [`learning_model_new_approaches.md`](../learning_model_new_approaches.md)
> (cited as **LMNA**; its requirements are R1–R8 and its validation targets
> V1–V18). It also draws on
> [`memory_model_analysis.md`](../memory_model_analysis.md) (**MMA**, findings
> M1–M13) and on the review of both,
> [`learning_model_docs_review.md`](../learning_model_docs_review.md) (findings
> X, D and N). The companion document
> [`essential_language_features.md`](essential_language_features.md) goes
> through five language features one at a time. The single measurement here (the
> cost of a nearest-neighbour table) is described in
> [Appendix A](#appendix-a--nearest-neighbour-benchmark).

---

## 0. Summary

**Short answer: no.** Neither is essential to reliable modelling in
`bilangsim`. Vector databases are not even useful at this scale. Embeddings are
an optional resource that helps with one narrow job: graded semantic
similarity.

| Component | Essential for reliable modelling? | Useful? | Where it could appear |
|---|---|---|---|
| **Vector database** (Milvus, Qdrant, pgvector, Chroma…) | **No** | **No.** The lexicon has at most ~60,000 concepts. An exact neighbour table is built once in ~30 s and takes 24 MB (measured, Appendix A). | nowhere |
| **Pretrained embeddings as an offline resource** (fastText, word2vec, transformer vectors) | **No** | Sometimes: graded semantic similarity at full-lexicon scale, building domain word lists where no corpus exists, checking the translation table | A1 (optional and gated, §6) |
| **Embeddings as the agent's knowledge** | **No** | Only inside the B and C research tracks, where they come with the learner | B (map weights), C (transformer weights) |
| **Distributed representations learned by a neural learner** | **No** for the five essential features at the level the research question needs. **Yes** if grammar must *emerge* from raw input. | research track | C |

**Why.** The project defines "reliable" through R1–R8: measurable outputs,
established memory findings, invariance to bookkeeping, identifiable
parameters, scale. **Embeddings help with none of these, and they hurt
identifiability.** Reliability comes from the shared foundations: a real
lexicon, grounded exposure, memory computed from timestamps, a battery of
virtual tests, and calibration. None of these uses embeddings.

**What embeddings contain.** An embedding is a compressed summary of
co-occurrence counts in a corpus (§2.1). `bilangsim`'s lexicon is a closed
list of tens of thousands of lemmas. At that size the *uncompressed*
statistics fit easily: per-setting frequency lists, co-occurrence tables and
WordNet relations. They are also easier to check against data, and they add
fewer free parameters. Reviewer finding N10 reached the same conclusion for
A1.

**The real choice** is between *specified* structure and *emergent* structure.
In approach A, structure is specified and testable. In B and C, it emerges,
which explains more but costs more. Embeddings come in with the second choice.
Whichever is chosen, reliability comes from the test infrastructure, and both
choices need it.

---

## 1. Where embeddings appear in the current plan

| Document | Role given to embeddings | Status |
|---|---|---|
| LMNA summary table | "give meaning, semantic fields and context (from stage A1)" in A; semantic input in B; learned internally in C | proposal |
| LMNA §3.2 | aligned fastText vectors for `ca` and `es`, used to cross-check the Apertium translation pairs and as the semantic input for B | proposal |
| LMNA §4.7 (A1) | cluster concepts into K semantic fields; give each setting a mix of fields; make inference from context depend on embedding similarity | proposal; **A0 uses no embeddings** |
| LMNA §5.2 (B) | concept vectors reduced to ~100 dimensions by PCA; the input to the semantic self-organising map | proposal |
| LMNA §6 (C) | learned inside each agent's transformer; no external embeddings | proposal |
| Review N10 | "the role of embeddings in A is overstated and not justified"; both A1 uses can be done with per-setting frequency lists and co-occurrence counts | review |
| Review N28, §4 | C has no grounding; B's adult meanings come from text-trained embeddings | review |

No document proposes a vector database. The question still matters, because
vector databases are standard parts of systems built on large language models,
including LLM "agents" with memory (§5.2).

---

## 2. What embeddings and vector databases are

### 2.1 Embeddings

An **embedding** here means a dense vector of about 100–1,000 numbers, learned
from a large corpus, that stands for a word or a word in context.

- **Static embeddings** (word2vec, GloVe, fastText) give one vector per word.
- **Contextual embeddings** (BERT-style models, the hidden states of large
  language models) give one vector per occurrence.

Words used in similar contexts get similar vectors.

**What they contain.** Static embeddings compress co-occurrence statistics.
Levy & Goldberg (2014) showed that word2vec's skip-gram with negative sampling
implicitly factorises a matrix of shifted pointwise mutual information (PMI)
between words and contexts. So the information in an embedding is already
present in co-occurrence counts. The embedding is a smoothed, lower-dimensional
version of those counts, and it generalises better when data are sparse.

**What they capture well.**
- Semantic *relatedness* and priming. Across a large semantic-priming dataset,
  prediction-based embeddings usually fit human behaviour better than
  count-based models (Mandera, Keuleers & Brysbaert, 2017).
- Meaning acquired by inference from text. Latent Semantic Analysis acquired
  vocabulary per paragraph of text at about the rate of schoolchildren reading.
  Most of its knowledge came from indirect inference, not from direct
  co-occurrence (Landauer & Dumais, 1997).

**What they capture poorly, for this project.**

| Limitation | Evidence | Why it matters here |
|---|---|---|
| Association is confused with similarity | SimLex-999 was built to separate the two; models did much worse on similarity than on relatedness (Hill, Reichart & Korhonen, 2015) | "cup" and "coffee" end up close, so semantic competition and interference come out wrong |
| No word form | Semantic vectors encode meaning, not spelling or sound. fastText's character n-grams mix in some form, which confounds the two. | Cognates and intelligibility depend on form (feature F2) |
| No grounding; adult, written meaning | Trained on Wikipedia or web text | A one-year-old's "dog" is not Wikipedia's "dog". This is the concern about adult meanings in B (review §4). |
| Unstable neighbours | Nearest neighbours change substantially under small changes to the corpus, more so for small corpora (Antoniak & Mimno, 2018) | Catalan corpora are much smaller than Spanish ones, so the two languages would get unequal noise |
| Cross-lingual alignment errors | Mapping two spaces onto each other suffers from "hubness": a few vectors are the nearest neighbour of too many others (Dinu, Lazaridou & Baroni, 2015) | Concept links between Catalan and Spanish need a dictionary anyway (LMNA §3.2) |

### 2.2 Vector databases

A **vector database** stores vectors and answers *approximate*
nearest-neighbour queries over millions to billions of items. It supports
metadata filters, incremental inserts and persistence. Examples are FAISS (a
library), Milvus, Qdrant, pgvector and Chroma. They are built for
retrieval-augmented generation, semantic search, recommendation and the memory
of LLM agents: a large collection that keeps growing, queried online with
unpredictable queries.

`bilangsim` has none of these needs. The lexicon is a fixed list. The
similarity structure is fixed for a run. Agents' knowledge is state that
evolves under exposure, not a collection to search.

---

## 3. Four roles embeddings could play in `bilangsim`

| Role | What it means | Proposed by | Essential? |
|---|---|---|---|
| **(a) Offline resource** | Fixed vectors shared by all agents, used to compute similarities or groupings before the run | LMNA A1 | No: explicit alternatives exist (§6) |
| **(b) Input encoding** | Vectors that feed a learner, e.g. B's semantic map | LMNA B | No: B can run on hand-built semantic features (feature norms, WordNet-derived features), as a robustness check |
| **(c) Agent knowledge** | The agent's lexicon *is* a set of weights (B's maps, C's transformer) | LMNA B, C | Only if one of those learners is chosen. It is a consequence of the learner, not a separate choice. |
| **(d) Episodic memory retrieved by similarity** | Each agent keeps past experiences as vectors and retrieves them by cosine similarity (the LLM-agent design, §5.2) | nobody | No, and it should be avoided (§5.2) |

---

## 4. Reliability check against R1–R8

| Requirement | Do embeddings help? | Comment |
|---|---|---|
| **R1** measurable units | no | Tests (CDI, vocabulary size, LexTALE, coverage) are scored on lemma lists from each agent's per-lemma state. |
| **R2** memory findings | no | The candidate memory rules (ACT-R, FSRS, the multiscale context model, PPE; review N4) keep a few numbers per item. |
| **R3** interaction of the two languages | no, slightly negative | Cognate effects depend on form; competition needs translation links from a dictionary. Semantic vectors mix meaning with partial form information (§2.1). |
| **R4** meaning and context | **yes, partly** | The only requirement where embeddings can contribute. Explicit alternatives exist (see F3 in the companion document). |
| **R5** invariance | neutral as a fixed resource; negative as agent state | SGD and online map learning depend on input order (review N30). |
| **R6** identifiability | **negative** | Corpus, model, dimension, alignment method, number of clusters K and the affinity function are all hidden degrees of freedom. Unstable neighbours add noise (§2.1). |
| **R7** scale | neutral if precomputed; strongly negative as agent state | B and C need GPUs (LMNA §5.8, §6.8). |
| **R8** interface | neutral | |

**Conclusion.** Embeddings can add *realism* for graded semantic effects. They
add nothing to *reliability* as the project defines it, and they work against
R6.

---

## 5. Vector databases: verdict

### 5.1 The lexicon is too small to need one

The largest lexicon in the plan is A4's ~60,000 concepts (LMNA §4.10). An
**exact** table of each concept's 50 nearest neighbours by cosine, computed by
brute force with NumPy on this machine:

| Concepts | Build time (once) | Embedding matrix (fp32, 300-d) | Neighbour table (top 50) |
|---|---|---|---|
| 10,000 | 1.0 s | 12 MB | 4 MB |
| 30,000 | 7.3 s | 36 MB | 12 MB |
| 60,000 | 31.7 s | 72 MB | 24 MB |

The table is computed once, stored in Parquet next to the lexicon, checksummed
in the run manifest (LMNA §3.9), and read as arrays at run time. A vector
database would add:
- a service to run;
- *approximate* answers, which break exact invariance (R5);
- per-query latency inside the hot loop (R7);
- one more component that has to be versioned.

It would add no capability.

### 5.2 LLM-agent memory is not a model of human lexical memory

"Generative agents" (Park et al., 2023) give each LLM agent a memory stream:
observations stored as text and embeddings, retrieved by a score that combines
recency, importance and embedding relevance. That design makes behaviour
believable in a sandbox. It was not built or validated as a model of how humans
learn and forget words. Using it in `bilangsim` would mean:
- no per-word knowledge in measurable units (R1);
- no link to the spacing, forgetting and age findings (R2);
- results that depend on prompts and on the language model's version (R5,
  R6);
- the cost of an LLM call per agent action (R7).

Even C does not need it. C's per-agent episodic buffer is a reservoir sampled
for replay (LMNA §6.2). If similarity search were ever wanted there, an
in-process FAISS flat index (exact search) would do.

**Verdict: vector databases have no role in `bilangsim` at any stage.**

---

## 6. Embeddings: verdict

### 6.1 Not essential

None of the five essential language features (companion document) needs
embeddings to reach the realism the research question requires:

| Feature | Without embeddings | Where embeddings would help |
|---|---|---|
| F1 Frequency-structured, growing lexicon | **fully** | nowhere |
| F2 Form overlap: cognates, false friends, language distance | **fully, and better** | nowhere (form, not meaning) |
| F3 Meaning, domains and context | **mostly** | graded semantic similarity where no explicit resource covers it; building domain word lists for settings with no corpus |
| F4 Bilingual lexical access, competition, code-switching | **fully** at word level | semantic competitors other than translations (second-order) |
| F5 Grammar and contact-induced change | **partly**: measurable contact phenomena yes, emergent grammar no | emergent grammar needs a neural learner (C), whose representations are learned, not imported |

### 6.2 Where embeddings are the right tool

They are worth using, offline, in four cases:

1. **Graded semantic similarity at full-lexicon scale** when a validation
   target needs it, such as semantic priming, or semantic interference in
   naming, over thousands of word pairs. Here embeddings are the best available
   predictor (Mandera et al., 2017).
2. **Building domain word lists for settings with no corpus**, e.g. work
   sectors or administration in Catalan. Start from seed words and expand
   through neighbours. This is a data-preparation tool, and its output must be
   checked by hand or against a small sample.
3. **Checking the translation table**: flag Apertium pairs whose aligned
   vectors disagree. This is already in LMNA §3.2.
4. **Inside B and C**, as part of the learner.

### 6.3 Where embeddings are the wrong tool

- **Form similarity** (cognates, intelligibility). Use edit distances on
  spelling and IPA (F2).
- **Translation equivalence and false friends.** These need a dictionary that
  says whether two forms share a concept.
- **Children's meanings.** Use CDI categories and child-directed co-occurrence
  (CHILDES).
- **Anything stored per agent in A.** A's knowledge is per-lemma memory
  traces. Review N10 asks for this to be stated plainly.

---

## 7. A gate for adding embeddings

Embeddings enter the model only if all five conditions hold:

| # | Condition |
|---|---|
| G1 | A named validation target needs graded semantic similarity. |
| G2 | No explicit resource (MCR wordnets, CDI categories, per-setting counts, PPMI co-occurrence) covers that need for the lemmas involved. Coverage is measured, not assumed. |
| G3 | The embedding variant beats the explicit variant on **held-out** targets (cf. the trivial-baseline principle, review N31). |
| G4 | Vectors are frozen and versioned, precomputed into a neighbour table, and checksummed in the run manifest. They are never queried at run time. |
| G5 | Results are checked for robustness across at least two embedding sources, or bootstrap samples of one (Antoniak & Mimno, 2018). |

---

## 8. An experiment that settles the question for A1

Instead of debating, run the same A1 learner with three semantic back-ends:

| Variant | P(concept \| setting) | Support from context for inferring a meaning |
|---|---|---|
| **S0: none** | age-specific register frequencies only (as in A0) | from coverage only |
| **S1: explicit** | per-setting frequency lists from register corpora, plus MCR domain labels where a corpus is missing | PPMI co-occurrence between the unknown concept and the recognised words |
| **S2: embeddings** | fields from clustering aligned fastText vectors, setting mixtures over fields (LMNA §4.7) | cosine similarity between the unknown concept and the context |

Score all three on the targets where semantics should matter:
- domain profiles: bilingual children's deficit is concentrated in home
  vocabulary, not school vocabulary (Bialystok et al., 2010; proposed V20);
- the effect of repetition on incidental learning (V15);
- the frequency and recognition-time patterns (V6), as a check that nothing
  else got worse.

**Decision rule.** Adopt S2 only if it beats S1 on held-out targets by more than
the run-to-run spread. Otherwise keep S1. Adopt S1 only if it beats S0 on V20.
Each step must earn its extra parameters.

---

## Appendix A — nearest-neighbour benchmark

Random unit vectors (the timing does not depend on the content), 300
dimensions, float32. Exact top-50 cosine neighbours, computed in blocks of 4,096
rows with NumPy matrix products and `argpartition`. Run with `uv run python` on
this branch's environment, WSL2, 14 logical cores.

```python
import time, numpy as np
rng = np.random.default_rng(0)
for n, dim, k in [(10_000, 300, 50), (30_000, 300, 50), (60_000, 300, 50)]:
    X = rng.standard_normal((n, dim), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    t0 = time.perf_counter()
    idx = np.empty((n, k), dtype=np.int32); sim = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, 4096):
        S = X[s:s+4096] @ X.T
        np.fill_diagonal(S[:, s:s+4096], -np.inf)
        part = np.argpartition(-S, k, axis=1)[:, :k]
        idx[s:s+4096] = part
        sim[s:s+4096] = np.take_along_axis(S, part, axis=1)
    print(n, time.perf_counter() - t0, X.nbytes / 1e6, (idx.nbytes + sim.nbytes) / 1e6)
```

## Appendix B — references

- Antoniak, M., & Mimno, D. (2018). Evaluating the stability of embedding-based word similarities. *TACL*, 6, 107–119.
- Bialystok, E., Luk, G., Peets, K. F., & Yang, S. (2010). Receptive vocabulary differences in monolingual and bilingual children. *Bilingualism: Language and Cognition*, 13(4), 525–531.
- Dinu, G., Lazaridou, A., & Baroni, M. (2015). Improving zero-shot learning by mitigating the hubness problem. *ICLR workshop track*.
- Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: evaluating semantic models with (genuine) similarity estimation. *Computational Linguistics*, 41(4).
- Landauer, T. K., & Dumais, S. T. (1997). A solution to Plato's problem: the latent semantic analysis theory of acquisition, induction and representation of knowledge. *Psychological Review*, 104(2), 211–240.
- Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. *NIPS 27*.
- Mandera, P., Keuleers, E., & Brysbaert, M. (2017). Explaining human performance in psycholinguistic tasks with models of semantic similarity based on prediction and counting. *Journal of Memory and Language*, 92, 57–78.
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: interactive simulacra of human behavior. *UIST '23*.
- Other references (Apertium, fastText aligned vectors, MCR, CHILDES, SUBTLEX) are listed in LMNA Appendix C and in the companion document.
