"""Core Five-Domains ontology for VetWelfare-NLP."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    NUTRITION = "nutrition"
    PHYSICAL_ENVIRONMENT = "physical_environment"
    HEALTH = "health"
    BEHAVIOURAL_INTERACTIONS = "behavioural_interactions"
    MENTAL_STATE = "mental_state"


@dataclass(frozen=True)
class Evidence:
    domain: Domain
    text: str
    start: int
    end: int
    valence: str = "negative"
    severity: str = "unknown"
    certainty: str = "asserted"
