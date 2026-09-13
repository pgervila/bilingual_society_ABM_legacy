# `agent.py` — agent hierarchy and the learning core

~2200 lines. The scientific heart of the model. Read
`../legacy_language_model_analysis.md` before changing any learning/memory logic.

## Module-level numba helpers (`agent.py:15-52`)

`@njit`-compiled hot-path math: `numba_comp_delta_S` (SM-2 stability increment),
`numba_comp_R` (retrievability), `numba_comp_max`, `numba_comp_exp`,
`numba_counter`, `numba_in1d` (a hand-rolled `in1d` used by the listen path).

## Class hierarchy

**Capability mixins:**
- `BaseAgent` (`:55`) — identity (`info`, `loc_info`, `lang_stats`), IC setup, growth, death, family lookups, `evolve`.
- `ListenerAgent(BaseAgent)` (`:394`) — `listen`, `update_lang_arrays`, `process_unknown_words`, `learn_unknown_words`, `update_words_memory`, `update_memory_retention`, school registration.
- `SpeakerAgent(ListenerAgent)` (`:762`) — `pick_vocab` (core speaking algo), `study_vocab`, `update_acquaintances`, friendship logic, exclusion reactions.
- `SchoolAgent(SpeakerAgent)` (`:1061`) — `go_to_school`, `speak_at_school`.
- `IndepAgent(SpeakerAgent)` (`:1112`) — grid-mobile agents; `move_to`, `speak_to_group`, `meet_agent`.

**Concrete age/role classes** (instantiated + scheduled):
`Baby` (`:1220`), `Child` (`:1316`), `Adolescent` (`:1376`), `Young` (`:1461`),
`YoungUniv` (`:1812`), `Adult` (`:1901`), `Worker` (`:1951`), `Teacher` (`:1956`),
`TeacherUniv` (`:2046`), `Pensioner` (`:2111`).

Agents **change class as they age** via `evolve(new_class, ...)` — `Child` →
`Adolescent` → `Young`/`YoungUniv` → `Adult` → `Pensioner`, and `Adult` →
`Teacher`/`TeacherUniv` on hire. `evolve` rebuilds the instance as the new class;
`StagedActivationModif.replace_agent` swaps it in the schedule in place.

`IndepAgent` subclasses receive their schedule index `ix_agent` in stage methods;
non-independent agents (`Baby`, `Child`) do not.

## Per-word memory state

Each of the four stores (`L1`, `L2`, `L12`, `L21`) in `self.lang_stats[lang]` holds
NumPy arrays of length `vocab_red`:

| Key | Shape | Meaning |
|---|---|---|
| `S` | `(vocab_red,)` float | Memory **stability** per word. |
| `t` | `(vocab_red,)` float | Steps elapsed since last activation. |
| `R` | `(vocab_red,)` float32 | **Retrievability** `= exp(-k·t/S)`, `k = ln(10/9)`. |
| `wc` | `(vocab_red,)` int64 | Cumulative lifetime activation count. |
| `pct` | `(max_life_steps,)` float | Per-age fraction of words with `R > 0.9` ("knowledge %"). |
| `excl_c` | `(max_life_steps,)` float | Per-age linguistic-exclusion counter. |
| `mem_eff` | `(vocab_red,)` int | Per-word min activations before encoding (`randint(3,7)`). |

## Initial conditions — **current state**

`_set_null_lang_attrs(lang, S_0=0.01, t_0=1000)` (`:114`) seeds a store with
`S=0.01`, `t=1000` (→ `R≈0`), `wc=0`, plus derived `pct`, `mem_eff`, `excl_c`.

`set_lang_ics(s_0=0.01, t_0=1000, biling_key=None)` (`:139`) currently calls
`_set_null_lang_attrs` for **all four stores** — i.e. graded initial bilingualism
is a **no-op**. The legacy per-age IC file was dropped; `biling_key` is accepted
for call-site compatibility but ignored. This is the documented deferred caveat.

## The learning core

### `update_lang_arrays(sample_words, mode_type='speak', delta_s_factor=0.1, ...)` (`:441`)

Entry point that turns a bag of `(word_indices, counts)` per language into memory
updates. Branches on `mode_type`:

- **`'listen'`/`'media'`/`'read'`** (`ds_factor = delta_s_factor`, default 0.1):
  1. Find words the agent already knows (`R > pct_threshold_und`).
  2. Increment `wc` for known-active words.
  3. For unknown active words, call `process_unknown_words` (normal) or `learn_unknown_words` (if `learning=True`) to decide which become eligible for a memory update.
- **`'speak'`** (`ds_factor = 1`): all spoken words are known by definition; increment `wc` directly.

Then, if any words remain, call `update_words_memory`.

### `process_unknown_words(...)` (`:517`)

Models catching new words. Two routes:
- **Recognition:** unknown word is *similar enough* (`edit_distances < max_edit_dist`) to a word already known in the correlated other language (`similarity_corr`) and that word's `R > pct_threshold_und`. Recognised words increment `wc`; those crossing `mem_eff` become updatable.
- **Identification:** otherwise, probability of picking up a new word scales with the fraction of the conversation understood (`min_prob_und` floor lets beginners learn). Drawn via `binomial`/`multinomial`; words crossing `mem_eff` become updatable.

### `update_words_memory(lang, act, act_c, ds_factor, ...)` (`:624`)

The SM-2 stability update. Constants `a=7.6, b=0.023, c=-0.031, d=-0.2`.
`ΔS = ds_factor · act_c · (a · S^(-b) · exp(c·100·R) + d)` via `numba_comp_delta_S`.
Applied **twice**: once with full `act_c`, then again with `act_c - 1` on the
updated `S` (the notebook's "simplification with good approx"). Then `t` for
activated words is reset to 0 via `step_mask`, and `update_memory_retention` recomputes `R`.

> **Note (candidate bug):** the double-application means total effective activations
> ≈ `2·act_c − 1` per step, roughly double the standalone IC-recipe path in
> `scripts/build_ics.py`. Flagged for the model-correctness follow-up.

### `update_memory_retention(lang, pct_threshold=0.9)` (`:679`)

Recomputes `R = exp(-k·t/S)` and updates `pct[age]` = fraction of words with
`R > pct_threshold`. Called both here and in the scheduler pre-stage bookkeeping.

## `pick_vocab(lang, conv_length='M', min_age_interlocs=None, ...)` (`:813`) — speaking + code-switch cascade

1. Sample `num_words` word *intents* from the age-indexed Zipf CDF (`cdf_data['s'][age]`, or the youngest interlocutor's age).
2. Pick the store variant: `L1` vs `L12` (or `L2` vs `L21`) by whichever has higher `pct`. **Known bug:** `L12`/`L21` `pct` is never updated (always 0), so the transition store is effectively never chosen at this step (`agent.py:876`).
3. **Retrieval mask:** a word is successfully spoken iff `random() <= R[word]`.
4. **Cascade for failed retrievals** (this is where code-switching emerges):
   - Missing words are retried in the paired store (`L1↔L12`, `L2↔L21`).
   - Still-missing words are retried in the far language; successes are added to the *transition* store (`max([lang,lang2], key=len)`), modelling creation/adaptation/translation.
5. `update_lang_arrays(spoken_words, delta_s_factor=1, ...)` — the speaker reinforces its own memory at full strength.

Returns `spoken_words` = `{store: [indices, counts]}`, consumed by listeners in `run_conversation`.

## Lifecycle methods

- **`evolve(new_class, ...)`** (`:254`): rebuild instance as `new_class`, carrying over state; subclasses override (e.g. `Adolescent.evolve` handles university enrolment).
- **`reproduce(...)`**: `Young` (`:1538`, `day_prob=0.0014`, ≤4 children) and `Adult` (`:1916`, `day_prob=0.0024`, `limit_age=40`) create babies; language-with-each-parent set via `get_newborn_lang` and the family network.
- **`random_death(...)`** (`:294`): draws against `model.death_prob_curve[age]`; overridden in `Teacher`/`TeacherUniv`/`Pensioner` to also handle job/course vacancy backfill.
- **`move_to`, `speak_to_group`, `meet_agent`** (`IndepAgent`): grid movement and group conversation initiation.

## The four stages (per concrete class)

Each concrete class implements `stage_1..stage_4`, called in shuffled order by the
scheduler. Behaviour is polymorphic:

| Class | Typical stage behaviour |
|---|---|
| `Baby` | Interacts with parents (`stage_2`/`stage_3` weighted toward mother, `prob_father=0.2`). |
| `Child` | Goes to school, speaks with family. |
| `Adolescent` | School + independent movement/socialising (`IndepAgent`). |
| `Young` / `YoungUniv` | Look for partners, jobs/university, reproduce, move to new home. |
| `Adult` / `Worker` | Jobs, family, reproduce (until 40). |
| `Teacher` / `TeacherUniv` | Teach classes (`speak_to_class`), speak with colleagues; custom death→backfill. |
| `Pensioner` | Gather family, socialise. |

Stage `num_days` arguments modulate how many of the 10 days in a step that
interaction type recurs.
