# `networks.py` — `NetworkBuilder`

Owns the four `networkx` graphs and per-step adjacency matrices. ~277 lines.

## The four graphs (`create_networks`, `networks.py:31`)

| Graph | Type | Edge meaning |
|---|---|---|
| `known_people_network` | `DiGraph` | Acquaintance. Edge attrs: `num_meet`, `lang` (language spoken with that person). Drives conversation-language stickiness in `get_conv_params`. |
| `friendship_network` | `Graph` | Friendships. |
| `family_network` | `DiGraph` | Kinship. Edge label = relation (`consort`, `father`/`child`, `sibling`, ...). |
| `jobs_network` | `Graph` | Random graph over `Job` instances (Erdős–Rényi, `p=0.3`). |

`family_mirror` (`:17`) maps each relation to its reciprocal (e.g. `father→child`,
`uncle→nephew`) so directed family edges can be set symmetrically.

## Setup

- `__init__` → `create_networks()` then `add_ags_to_networks(schedule.agents)` (adds all agents as nodes to the three agent graphs).
- `build_networks()` (`:53`) → `define_family_networks()` → `define_friendship_networks()` → `define_jobs_network()`.

### `define_family_networks` (`:77`)
Partitions each cluster's agents into groups of `family_size` (4) and calls
`define_family_links` per group. Agents left over (cluster size not divisible by 4)
just get `set_lang_ics()` and no family edges. Finally calls
`geo.assign_school_jobs()` so a whole family relocates together if needed.

Marriage combinations are restricted to linguistically compatible pairs (`0-1`,
`1-1`, `1-2`).

### `define_family_links` (`:58`)
Resolves interaction languages via `model.get_lang_fam_members(family)`, then wires
consort↔consort, children↔father, children↔mother, sibling↔sibling edges in both
`family_network` and `known_people_network` via `set_link_with_relatives`.

### `define_friendship_networks(init_max_num_friends=10)` (`:99`)
For each agent, pulls candidate peers from its occupation context (job `employees`
for `Adult`/`Young`/`Teacher`; school course `students` for `Child`/`Adolescent`),
and calls `make_friend` for those passing `check_friend_conds` (language-distance +
capacity checks), up to `min(max_num_friends, init_max_num_friends)`.

### `define_jobs_network(p=0.3)` (`:130`)
`nx.gnp_random_graph` over all `Job` instances, relabelled to the job objects.

## Runtime methods

- **`set_newborn_family_links(agent, father, mother, lang_with_father, lang_with_mother)`** (`:138`): wires a newborn into the family graph (parents, siblings, and extended relatives via `family_mirror`).
- **`set_link_with_relatives(agents, relatives, labels, lang_with_relatives=None)`** (`:186`): generic directed-edge writer that also mirrors the reciprocal relation.
- **`compute_adj_matrices()`** (`:222`): builds adjacency matrices for family + friendship networks, snapshotted once per step and held constant across stages. **Modernization fix:** wrapped with `csr_matrix(...)` because `nx.adjacency_matrix` now returns a `csr_array`/`coo_array` lacking `.indices`.
- **`add_ags_to_networks(ags, *more_ags)`** (`:41`): add nodes to the three agent graphs.

## Persistence

`__getstate__`/`__setstate__` (`:255`) exclude the cached adjacency matrices from the
pickle (they are recomputed each step), keeping dill snapshots smaller.

## Plotting helpers

`plot_family_networks`, `plot_jobs_network` — matplotlib visualisations (analysis-only).
