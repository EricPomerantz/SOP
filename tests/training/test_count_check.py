from training.count_check import check, expected_counts

SEQ = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FULL_BAYS = {"chassis": 1, "cover": 0, "psu": 0, "psu_bay_empty": 2}


def test_expected_during_third_fan():
    e = expected_counts(SEQ, 4)
    assert (e.fan, e.fan_bay_empty, e.psu, e.cover, e.chassis) == ((2, 3), (3, 4), (0, 0), (0, 0), (1, 1))


def test_expected_before_any_step_and_during_psus():
    assert expected_counts(SEQ, 1).fan == (0, 0) and expected_counts(SEQ, 1).fan_bay_empty == (6, 6)
    e = expected_counts(SEQ, 9)  # second PSU
    assert (e.fan, e.psu, e.psu_bay_empty) == ((6, 6), (1, 2), (0, 1))


def test_cover_hides_parts():
    assert expected_counts(SEQ, 10).fan is None and expected_counts(SEQ, 10).cover == (0, 1)
    assert expected_counts(SEQ, 11).cover == (1, 1) and expected_counts(SEQ, 11).psu is None


def test_check_reasons_and_hand_allowance():
    e = expected_counts(SEQ, 4)
    assert check({"fan": 4, "fan_bay_empty": 2, **FULL_BAYS}, e, hands=0).reasons[0] == "fan: 4 not in 2–3"
    assert check({"fan": 1, "fan_bay_empty": 3, **FULL_BAYS}, e, hands=1).ok


def test_hands_never_excuse_chassis_or_cover():
    e = expected_counts(SEQ, 4)
    result = check({"fan": 2, "fan_bay_empty": 4, **FULL_BAYS, "chassis": 0}, e, hands=2)
    assert not result.ok and result.reasons == ["chassis: 0 not in 1–1"]


def test_missing_class_counts_as_zero():
    assert not check({}, expected_counts(SEQ, 1), hands=0).ok


def test_hands_relax_lower_bound_by_one_each_not_entirely():
    e = expected_counts(SEQ, 4)  # fans 2–3
    assert check({"fan": 0, "fan_bay_empty": 3, **FULL_BAYS}, e, hands=1).reasons == ["fan: 0 not in 2–3"]
    assert check({"fan": 0, "fan_bay_empty": 3, **FULL_BAYS}, e, hands=2).ok
