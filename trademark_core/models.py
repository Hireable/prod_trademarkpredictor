"""
Single source of truth (SSoT) for all data models in the Trademark-AI system.

This module defines the core Pydantic models used across the API layer, domain logic,
and tests. These models serve as the canonical schema definitions and should never
be redeclared elsewhere in the codebase.
"""

from typing import Literal, List, Annotated
from pydantic import BaseModel, Field, conint


class Mark(BaseModel):
    """A trademark mark, consisting of text and registration details."""
    wordmark: str = Field(..., description="Literal mark text, case-sensitive")
    is_registered: bool = False
    registration_number: str | None = None


class GoodService(BaseModel):
    """A good or service classification under the Nice Agreement."""
    term: str
    nice_class: Annotated[int, Field(ge=1, le=45)]


# Comparison result enums
EnumStr = Literal["dissimilar", "low", "moderate", "high", "identical"]


class MarkComparison(BaseModel):
    """Comparison results between two trademarks across multiple dimensions."""
    visual: EnumStr
    aural: EnumStr
    conceptual: EnumStr
    overall: EnumStr


class GoodServiceComparison(BaseModel):
    """Comparison result between two goods/services terms."""
    similarity: EnumStr


class CasePrediction(BaseModel):
    """Complete trademark opposition case prediction including all comparisons."""
    mark_comparison: MarkComparison
    goods_services_comparisons: List[GoodServiceComparison]
    likelihood_of_confusion: bool
    opposition_outcome: dict  # {result:str, confidence:float, reasoning:str} 