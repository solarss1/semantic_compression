from src.roi.class_priorities import parse_class_weights, DEFAULT_CLASS_WEIGHTS


def test_parse_explicit_weights():
    cfg = {"class_weights": {1: 1.0, 3: 0.5}}
    w = parse_class_weights(cfg)
    assert w[1] == 1.0
    assert w[3] == 0.5


def test_parse_priority_tiers():
    cfg = {
        "class_priority": {"high": [1], "low": [62]},
        "priority_values": {"high": 1.0, "low": 0.3},
    }
    w = parse_class_weights(cfg)
    assert w[1] == 1.0
    assert w[62] == 0.3


def test_default_weights_nonempty():
    assert 1 in DEFAULT_CLASS_WEIGHTS
    assert DEFAULT_CLASS_WEIGHTS[1] == 1.0
