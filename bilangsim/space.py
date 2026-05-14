"""Minimal grid replacement for the dropped Mesa dependency.

Provides only the MultiGrid surface bilangsim actually uses:
place_agent, move_agent, _remove_agent, get_cell_list_contents, width, height.
"""


class MultiGrid:
    """A 2D grid where each cell holds a set of agents.

    Signature is (width, height, torus) to match mesa.space.MultiGrid. Note that
    model.py historically calls MultiGrid(height, width, False) with
    width == height == 100, so the argument swap is harmless and is kept as-is.
    """

    def __init__(self, width, height, torus=False):
        self.width = width
        self.height = height
        self.torus = torus
        self._cells = {}

    def _cell(self, pos):
        if pos not in self._cells:
            self._cells[pos] = set()
        return self._cells[pos]

    def place_agent(self, agent, pos):
        self._cell(pos).add(agent)
        agent.pos = pos

    def move_agent(self, agent, pos):
        old = getattr(agent, 'pos', None)
        if old is not None and old in self._cells:
            self._cells[old].discard(agent)
        self.place_agent(agent, pos)

    def _remove_agent(self, pos, agent):
        if pos in self._cells:
            self._cells[pos].discard(agent)
        if getattr(agent, 'pos', None) == pos:
            agent.pos = None

    def get_cell_list_contents(self, cell_list):
        """Accept a single (x, y) position or an iterable of positions."""
        if (len(cell_list) == 2 and
                all(isinstance(c, int) for c in cell_list)):
            cell_list = [cell_list]
        contents = []
        for pos in cell_list:
            contents.extend(self._cell(pos))
        return contents

    def __getitem__(self, x):
        """Support mesa-style ``grid[x][y]`` indexing, returning the cell's agent set."""
        return _GridColumn(self, x)


class _GridColumn:
    """Thin proxy so that ``grid[x][y]`` resolves to the agent set at (x, y)."""

    def __init__(self, grid, x):
        self._grid = grid
        self._x = x

    def __getitem__(self, y):
        return self._grid._cell((self._x, y))
