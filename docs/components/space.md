# `space.py` — vendored `MultiGrid`

Minimal 2D grid that replaces `mesa.space.MultiGrid`. ~66 lines. Provides only the
surface `bilangsim` actually uses. Each cell holds a **set** of agents.

## `MultiGrid(width, height, torus=False)` (`space.py:8`)

> **Argument-order note:** `BiLangModel.__init__` calls `MultiGrid(height, width, False)`
> with `width == height == 100`, so the swap is harmless and kept as-is to match the
> legacy call site.

State: `width`, `height`, `torus`, `_cells` (dict `pos → set`).

| Method | Behaviour |
|---|---|
| `_cell(pos)` | Lazily create + return the set at `pos`. |
| `place_agent(agent, pos)` | Add agent to cell set; set `agent.pos = pos`. |
| `move_agent(agent, pos)` | Discard from old cell, then `place_agent`. |
| `_remove_agent(pos, agent)` | Discard from cell; set `agent.pos = None` if it matched. |
| `get_cell_list_contents(cell_list)` | Accepts a single `(x,y)` **or** an iterable of positions; returns a flat list of agents. |
| `__getitem__(x)` | Returns a `_GridColumn` proxy so `grid[x][y]` indexing works. |

## `_GridColumn` (`space.py:58`)

Proxy returned by `MultiGrid.__getitem__`; its `__getitem__(y)` returns the cell set
at `(x, y)`. Added during modernization to support the legacy `grid[x][y]` access
pattern that Mesa provided.

## Not implemented

Neighbourhood queries, torus wraparound math, coordinate iteration helpers — none of
these are used by `bilangsim`, so they are intentionally absent.
