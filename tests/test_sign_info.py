from sign_info import GTSRB_LABELS, get_sign_info


def test_standard_gtsrb_has_43_classes():
    assert len(GTSRB_LABELS) == 43


def test_30_sign_is_speed_limit_not_zone_30():
    title, warning = get_sign_info(1, "Speed limit (30km/h)")
    assert "30" in title
    assert "30" in warning
    assert "زون" not in warning


def test_stop_warning():
    title, warning = get_sign_info(14, "Stop")
    assert "STOP" in title
    assert "توقف" in warning
