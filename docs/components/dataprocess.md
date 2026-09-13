# `dataprocess.py` — data collection, persistence, visualisation

Standalone classes (formerly subclassed Mesa `DataCollector`). ~400 lines.
Storage is **Parquet** for tabular results + **dill** for full-model snapshots —
no HDF5/deepdish/tables anywhere.

## `DataProcessor` (`dataprocess.py:13`)

### Reporters (`__init__`)
Two dicts of lambdas evaluated each collected step.

**Model-level** (`model_reporters`): `pct_spa`/`pct_bil`/`pct_cat` (population
fractions by language type via `get_lang_stats`), `total_num_agents`,
`pct_cat_in_biling` (`get_global_bilang_inner_evol`).

**Agent-level** (`agent_reporters`): per-agent `pct_cat_knowledge`,
`pct_spa_knowledge` (from `lang_stats[...]['pct'][age]`), `pct_L21_knowledge`/
`pct_L12_knowledge` (mean `R` of the transition stores), `tokens_per_step_spa`/`_cat`
(`wc_final - wc_init` sum), `x`/`y`, `age`, `language`, `excl_c` (exclusion counter
in the *other* language), `clust_id`, `agent_type`, `num_conv_step`.

`model_vars`/`agent_vars` accumulate collected values in memory until flushed.
`init_conds` captures the reproducible setup metadata (cluster geometry, grid size,
`init_lang_distrib`, sort flags).

### `collect()` (`:53`)
Appends one row of every model reporter and, per agent, `(unique_id, value)` tuples
for every agent reporter. Called by `BiLangModel.step()` (not by the scheduler, so
warmup is excluded).

### DataFrame builders
- `get_model_vars_dataframe()` — one column per model var, index = tick range.
- `get_agent_vars_dataframe()` — MultiIndex `(Step, AgentID)`, one column per agent var.

### Persistence
- **`save_model_data(save_data_freq)`** (`:239`): write `model_data_part{part:05d}.parquet` and `agent_data_part{part:05d}.parquet` (`part = steps // save_data_freq`); on the first flush also dill-dump `init_conds.pkl` (holds numpy arrays, so not Parquet-friendly). Then **clears** `model_vars`/`agent_vars` to free RAM.
- **`load_model_data(save_dir='')`** (`:260`, staticmethod): glob + `pd.concat` all Parquet parts; returns `(model_df, agent_df)`, either possibly `None`.
- **`pickle_model()` / `unpickle_model(filename)`** (`:197`/`:202`): full-model dill snapshot (serialises the whole agent + network object graph).

### Query/analysis helpers
`get_agent_by_id`, `get_agents_by_type`, `get_tokens_per_step`,
`get_tokens_stats_per_type`, `get_lang_stats(lang_type)` (population fraction whose
dominant language is `lang_type`), `get_global_bilang_inner_evol`,
`get_agents_attrs_value`, `get_attrs_avg_map`.

## `DataViz` (`:273`)

Live/at-run visualisation. `show_results(ag_attr='language', step=None, ...)` renders
agent attributes on the grid (used by `run_model`'s `viz_steps_period` and
`run_and_animate`).

## `VizImpData` (`:336`)

`__init__(save_dir='')` loads saved Parquet parts via `DataProcessor.load_model_data`;
`show_imported_results` displays them. (Modernized off the old HDFStore reader.)

## `PostProcessor` (`:345`)

Offline analysis over saved results. `__init__(data_filename=None, save_dir='')`
loads `model_data`/`agent_data` (Parquet) and `init_conds.pkl` if present.
Methods: `ag_results_by_id`, `ag_results_by_type`, `get_grouped_lang_stats`,
`plot_lang_trend`, `get_num_tokens_per_agent_type`,
`group_by_step_and_agent_type`/`group_by_step_and_language_type`,
`plot_population_size`.

> **Caveat:** `plot_population_size` uses pandas-2.x-removed `.count(level=...)` — an
> untested analysis helper, not run-blocking. Flagged for later.
