# `geomapping.py` — `GeoMapper`

Lays out the city on the grid and creates the initial population, then handles
runtime spatial bookkeeping. ~592 lines.

## Entry point: `map_model_objects()` (`geomapping.py:40`)

Called from `BiLangModel.__init__`. Orchestrates the full setup:
1. `compute_cluster_centers` — place `num_clusters` cluster centres on the grid, respecting `min_dist` and grid margins.
2. `compute_cluster_sizes` — assign a population size to each cluster (min 100).
3. `set_clusters_info` — build the per-cluster `clusters_info[idx]` dict (centre, size, containers for agents/homes/schools/jobs/university).
4. `map_jobs`, `map_schools`, `map_universities` (`pct_univ_towns=0.2`), `map_homes` (`num_people_per_home=4`) — place city objects.
5. `generate_langs_per_clust` then `map_lang_agents` — create the population.

## Cluster layout

- **`compute_cluster_centers(min_dist=0.10, ...)`** (`:48`): random centres with a minimum pairwise distance, kept within `[min_grid_pct_val, max_grid_pct_val]` of the grid.
- **`compute_cluster_sizes(min_size=100)`** (`:97`): partition `num_people` across clusters.
- **`generate_points_coords` / `sort_coords_in_clust`**: scatter individual agent positions around each centre; optionally sort by distance-to-origin.

## Population generation

### `generate_langs_per_clust()` (`:370`)
Draws a length-`num_people` array of language labels `{0,1,2}` from
`init_lang_distrib`, then splits it across clusters by cumulative cluster sizes.
Sorting behaviour depends on two model flags:
- `lang_ags_sorted_by_dist` — sort the whole array (segregates languages by cluster/distance).
- `lang_ags_sorted_in_clust` — sort within each cluster.
- When neither full sort is requested, labels are still grouped into blocks of 4 (then blocks shuffled) so each family-of-4 is linguistically coherent.

Result stored in `langs_per_clust` (list of per-cluster label arrays).

### `map_lang_agents(parents_age_range=(25,50), children_age_range=(2,17), max_age_diff=35)` (`:414`)
For each cluster, iterate the language labels in groups of `family_size` (4):
- `create_new_family(...)` → 2 parents + 2 children with sampled ages,
- assign a shared `Home`,
- add to `clust_info['agents']`, the grid, and the schedule (**not** yet to networks),
- parents `get_job(keep_cluster=True, move_home=False)`,
- children assigned to the nearest `School` (course assigned later).

Leftover agents (cluster size not divisible by 4) are created as lone `Adult`s with
a home and job.

### `create_new_family` / `create_new_agent` (`:298` / `:282`)
Instantiate concrete agent classes with sampled sex/age; `create_new_agent` pulls a
unique id from `model.set_available_ids`.

### `assign_school_jobs()` (`:474`)
Called by `NetworkBuilder` after families are wired. Runs `School.set_up_courses()`
for every school (groups students by age, hires teachers). Asserts every course got
a teacher — otherwise raises a "school lang policy cannot be met by current
population" error.

## Runtime spatial bookkeeping

- **`add_agents_to_grid_and_schedule(ags)`** (`:500`): place on grid + `schedule.add`.
- **`update_agent_clust_info(agent, curr_clust, update_type='remove'|'add', ...)`** (`:516`): move an agent between clusters' `agents` sets when it relocates.
- **`add_new_home(clust_id)` / `add_new_school(clust_id)`** (`:345`/`:357`): create objects on demand (e.g. when immigration or growth exceeds capacity).

## Query helpers

`get_clusters_with_univ`, `get_current_clust_size`, `get_lang_distrib_per_clust`,
`get_dominant_lang_per_clust` — used for placement decisions and language-policy logic.

## Placeholders

`import_geo` / `import_model_ics` (`:29`/`:36`) are legacy stubs from when geography
and ICs were loaded from files; they are effectively no-ops in the current runtime-
generation flow.
