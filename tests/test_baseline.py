from vetwelfare_nlp.baseline import extract_evidence, profile_text
from vetwelfare_nlp.ontology import Domain


def test_extracts_health_and_nutrition() -> None:
    text = "The dog has reduced appetite and is painful on hip extension."
    domains = {item.domain for item in extract_evidence(text)}
    assert Domain.NUTRITION in domains
    assert Domain.HEALTH in domains


def test_simple_negation() -> None:
    assert not extract_evidence("No vomiting was reported.")


def test_abstains_without_evidence() -> None:
    result = profile_text("Routine follow-up visit.")
    assert result["mental_state"]["abstain"] is True
