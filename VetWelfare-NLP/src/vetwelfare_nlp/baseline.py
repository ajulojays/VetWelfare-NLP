"""Transparent lexicon baseline for welfare-signal extraction."""

from __future__ import annotations

import re
from collections import defaultdict

from .ontology import Domain, Evidence

LEXICON: dict[Domain, tuple[str, ...]] = {
    Domain.NUTRITION: (
        "reduced appetite", "poor appetite", "anorexia", "not eating",
        "weight loss", "dehydrated", "thirst", "body condition",
    ),
    Domain.PHYSICAL_ENVIRONMENT: (
        "heat stress", "cold exposure", "poor ventilation", "overcrowded",
        "inadequate shelter", "wet bedding", "noise", "confined",
    ),
    Domain.HEALTH: (
        "pain", "painful", "lameness", "wound", "injury", "vomiting",
        "diarrhea", "dyspnea", "fever", "infection", "pruritus",
    ),
    Domain.BEHAVIOURAL_INTERACTIONS: (
        "reluctant to rise", "withdrawn", "aggressive", "fearful", "anxious",
        "stereotypy", "not playing", "social isolation", "avoids handling",
    ),
}

NEGATION = re.compile(r"\b(no|not|without|denies|negative for)\b", re.IGNORECASE)


def extract_evidence(text: str) -> list[Evidence]:
    """Return matched, non-negated evidence spans from a clinical narrative."""
    findings: list[Evidence] = []
    lower = text.lower()
    for domain, terms in LEXICON.items():
        for term in terms:
            for match in re.finditer(re.escape(term), lower):
                left_context = text[max(0, match.start() - 30):match.start()]
                if NEGATION.search(left_context):
                    continue
                findings.append(
                    Evidence(
                        domain=domain,
                        text=text[match.start():match.end()],
                        start=match.start(),
                        end=match.end(),
                    )
                )
    return sorted(findings, key=lambda item: item.start)


def profile_text(text: str) -> dict[str, object]:
    """Create a simple evidence-linked welfare profile."""
    evidence = extract_evidence(text)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        grouped[item.domain.value].append(item.text)

    profile: dict[str, object] = {}
    for domain in list(Domain)[:4]:
        spans = grouped.get(domain.value, [])
        profile[domain.value] = {
            "score": min(1.0, len(spans) / 2),
            "evidence": spans,
            "documented": bool(spans),
        }

    negative_count = sum(bool(grouped.get(domain.value)) for domain in list(Domain)[:4])
    if negative_count == 0:
        mental = {"label": "insufficient_evidence", "confidence": 0.0, "abstain": True}
    else:
        mental = {
            "label": "likely_negative",
            "confidence": round(min(0.95, 0.45 + 0.12 * negative_count), 2),
            "abstain": negative_count < 2,
        }
    profile[Domain.MENTAL_STATE.value] = mental
    return profile
