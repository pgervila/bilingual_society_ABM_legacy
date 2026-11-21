# Modelling solutions

> **Status (Nov 2026):** analysis only. No code was changed. This folder
> collects conclusions about *how* to model realistic learning and realistic
> language in `bilangsim`. It builds on three documents:
> - [`memory_model_analysis.md`](../memory_model_analysis.md) (MMA): the legacy
>   memory rule, findings M1–M13;
> - [`learning_model_new_approaches.md`](../learning_model_new_approaches.md)
>   (LMNA): shortcomings S1–S17, approaches A/B/C;
> - [`learning_model_docs_review.md`](../learning_model_docs_review.md): a
>   review of both.

## Questions answered here

1. Are **embeddings** and **vector databases** essential for reliable modelling?
2. Which **language features** are essential for realistic learning and
   realistic language? Can each be captured without embeddings?

## Short answers

**Vector databases: no.** They have no role at any stage of any approach. The
largest planned lexicon has ~60,000 concepts. An exact table of nearest
neighbours is built once in ~30 s and takes 24 MB (measured). A vector database
would add a service, approximate answers and latency, and no capability.

**Embeddings: not essential.**
- Reliability as the project defines it (R1–R8: measurable outputs, memory
  findings, invariance, identifiability, scale) comes from the shared
  foundations. None of these uses embeddings: a real lexicon, grounded
  exposure, memory computed from timestamps, virtual tests, calibration.
- Embeddings are compressed co-occurrence statistics. For a closed lexicon of
  tens of thousands of lemmas, the explicit statistics are affordable, more
  interpretable and easier to test.
- Their only useful role is an offline source of graded semantic similarity,
  behind a five-condition gate.

**Five essential language features:**

| # | Feature | Without embeddings? |
|---|---|---|
| F1 | Frequency-structured, growing lexicon (frequency, contextual diversity, burstiness, coverage) | **fully** |
| F2 | Form overlap between the languages: cognates, false friends, distance between the languages | **fully, and better** |
| F3 | Meaning, domains and context (domain-specific vocabulary, inference from context) | **mostly**: embeddings optional, offline, gated |
| F4 | Bilingual lexical access: comprehension vs production, competition, control, code-switching | **fully** at word level |
| F5 | Grammar and contact-induced change: loans, calques, structural convergence | **partly**: measurable contact phenomena yes, emergent grammar no |

**The real choice is not "embeddings or not".** It is *specified* structure,
which is testable and cheap (approach A), against *emergent* structure, which
explains more but costs more (B, C). Embeddings come with the second choice.
The one feature an embedding-free design cannot fully capture is grammar
emerging from raw input. That needs a neural learner (C), whose representations
are learned rather than imported.

## Files

| File | Content |
|---|---|
| [`embeddings_and_vector_databases.md`](embeddings_and_vector_databases.md) | What embeddings and vector databases are and contain; four roles they could play; a check against R1–R8; verdicts; a gate for adding embeddings; an experiment that settles the question for A1; the neighbour-table benchmark |
| [`essential_language_features.md`](essential_language_features.md) | The five features: why each is essential, what realism requires, an embedding-free mechanism, data sources, a verdict. Also: realistic learning across features, what was left out, an embedding-free layered design, new validation targets V19–V23, and implications for the LMNA plan. |

## Suggested changes to the plan in LMNA

1. Make A1's semantics explicit by default: per-setting frequencies, MCR domain
   labels, optional PPMI. Make embeddings a gated variant.
2. Specify the form layer: spelling and IPA distances, a false-friend list, one
   cognate path at first.
3. Add a contact-and-grammar stage: lexical variants plus 5–10 grammatical
   variables, with targets V21–V23. Add an acceptability module to the online
   study now, because collecting data takes the longest.
4. Run B0 with both embedding and hand-built semantic input, to test whether
   B's results depend on the embeddings.
5. Keep C as the only route to emergent grammar, with no vector database.
