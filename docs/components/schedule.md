# `schedule.py` — `StagedActivationModif`

The scheduler. Standalone class (formerly subclassed Mesa `StagedActivation`);
fully overrides `step()`. ~96 lines.

## Construction (`schedule.py:13`)

`StagedActivationModif(model, stage_list, shuffle=False, shuffle_between_stages=False)`.
`BiLangModel` builds it with `stage_list=["stage_1","stage_2","stage_3","stage_4"]`
and `shuffle=True`. Tracks `steps`, `time`, `agents` (a list), `stage_time = 1/len(stage_list)`.

Trivial bookkeeping: `add`, `remove` (removes all occurrences), `get_agent_count`.

## `step(pct_threshold=0.9)` — anatomy of one step (`schedule.py:33`)

Runs once per model step (≈10 simulated days). Order:

### 1. Pre-stage bookkeeping (every agent)
For each agent:
- `_conv_counts_per_step = 0` (reset conversation counter).
- `grow()` — age += 1 (clamped to `max_life_steps - 1`).
- For each of the four langs:
  - snapshot `wc_init[lang] = wc.copy()`,
  - `update_word_activation_elapsed_time(lang)` — age the `t` array,
  - `update_memory_retention(lang, pct_threshold)` — recompute `R` and `pct[age]`,
  - `reset_step_mask(lang)`.
- `update_lang_switch()` — reclassify the agent as mono/bilingual from current knowledge.

Then, if `shuffle`, shuffle the agent list.

### 2. Adjacency snapshot
`model.nws.compute_adj_matrices()` — network adjacency matrices are held **constant
through all four stages** of this step.

### 3. Four stages
For each `stage` in `stage_list`, for each agent (in list order):
- `IndepAgent` subclasses → `getattr(ag, stage)(ix_ag)` (passed their schedule index),
- others → `getattr(ag, stage)()`.

`self.time += stage_time` after each stage.

### 4. Post-stage: reproduction + death
Iterate a **shallow copy** of the agent list (agents may be removed mid-iteration):
- snapshot `wc_final[lang] = wc.copy()`,
- if `isinstance(ag, Young)` → `ag.reproduce()`,
- `ag.random_death()`.
Wrapped in `try/except KeyError` because an agent may have been replaced (e.g. hired
as a `Teacher` to backfill a dead one) and no longer be resolvable.

### 5. Periodic hooks
- **Every 2 years** (`steps % (2*steps_per_year) == 0`): `model.update_friendships()`.
- **Every year** (`steps and steps % steps_per_year == 0`): `model.update_centers()` (roll over school/university courses + job language policies), and if `model.pct_immigration`, `model.add_immigration()`.

Finally `self.steps += 1`.

## `replace_agent(old_agent, new_agent)` (`schedule.py:92`)

Swaps `new_agent` into `old_agent`'s exact position in the schedule list. Used by
`evolve` so an aging/hired agent keeps its slot.

## Note on data collection

`schedule.step()` does **not** collect data. `BiLangModel.step()` calls
`schedule.step()` then `data_process.collect()`. The warmup phase calls
`schedule.step()` directly, which is why warmup steps are excluded from collected data.
