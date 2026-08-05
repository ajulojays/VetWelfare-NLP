from vetwelfare_nlp.vetjudge import _jaccard, _shannon_entropy, _stable_blind_id, _tokenize_evidence


def test_blind_id_is_stable_and_salted():
    assert _stable_blind_id("model-a", "salt-1") == _stable_blind_id("model-a", "salt-1")
    assert _stable_blind_id("model-a", "salt-1") != _stable_blind_id("model-a", "salt-2")
    assert _stable_blind_id("model-a", "salt-1").startswith("model_")


def test_entropy_distinguishes_agreement_from_disagreement():
    assert _shannon_entropy(["negative"] * 5) == 0.0
    assert _shannon_entropy(["negative", "positive"]) == 1.0


def test_evidence_tokenization_and_overlap():
    left = _tokenize_evidence("Painful on hip extension")
    right = _tokenize_evidence("hip extension painful")
    assert left == right
    assert _jaccard(left, right) == 1.0
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"pain"}, {"itch"}) == 0.0
