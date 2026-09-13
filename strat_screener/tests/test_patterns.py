from strat_screener.patterns import detect_pattern


def test_2_1_2_continuation_up():
    types = ["2u", "2u", "1", "2u"]
    m = detect_pattern(types, 3)
    assert m.name == "2-1-2 continuation"
    assert m.direction == "up"
    assert m.n_bars == 3


def test_2_1_2_reversal_down():
    types = [None, "2u", "1", "2d"]
    m = detect_pattern(types, 3)
    assert m.name == "2-1-2 reversal"
    assert m.direction == "down"


def test_3_1_2():
    types = [None, "3", "1", "2u"]
    m = detect_pattern(types, 3)
    assert m.name == "3-1-2"
    assert m.direction == "up"


def test_1_2_2_continuation():
    types = [None, "1", "2d", "2d"]
    m = detect_pattern(types, 3)
    assert m.name == "1-2-2 continuation"
    assert m.direction == "down"


def test_2_2_reversal():
    types = [None, "1", "2u", "2d"]
    m = detect_pattern(types, 3)
    assert m.name == "2-2 reversal"
    assert m.direction == "down"


def test_no_pattern_on_inside_bar():
    types = [None, "2u", "2u", "1"]
    assert detect_pattern(types, 3) is None


def test_no_pattern_on_two_bar_continuation():
    # 2u then 2u is just continuation of the same directional push, not a
    # recognized combo by itself.
    types = [None, "2u", "2u"]
    assert detect_pattern(types, 2) is None


def test_out_of_range_index():
    assert detect_pattern(["1", "2u"], 0) is None
    assert detect_pattern(["1", "2u"], 5) is None
