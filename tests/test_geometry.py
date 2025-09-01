from thermal_api.utils.geometry import top_third


def test_top_third():
    y1, y2 = top_third(10, 40)
    assert y1 == 10 and (y2 - y1) == 10 # 全体30の上1/3