# tests/test_space.py
from bilangsim.space import MultiGrid


class _Ag:
    pos = None


def test_place_move_remove_and_contents():
    grid = MultiGrid(10, 10, False)
    a, b = _Ag(), _Ag()
    grid.place_agent(a, (2, 3))
    grid.place_agent(b, (2, 3))
    assert set(grid.get_cell_list_contents((2, 3))) == {a, b}
    assert a.pos == (2, 3)
    grid.move_agent(a, (4, 5))
    assert grid.get_cell_list_contents((2, 3)) == [b]
    assert set(grid.get_cell_list_contents((4, 5))) == {a}
    grid._remove_agent((4, 5), a)
    assert grid.get_cell_list_contents((4, 5)) == []
    assert a.pos is None
    assert grid.width == 10 and grid.height == 10
