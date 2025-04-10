# src/tools/similarity_tools.py
"""
ADK Tools for calculating various trademark similarity scores.
"""

from typing import List, Optional

from google.adk.tools.toolset import Toolset
from pydantic import BaseModel, Field

from src.models import Wordmark, GoodsService # Import Pydantic models for input types
from src.similarity import (
    calculate_visual_similarity,
    calculate_goods_services_similarity,
    calculate_aural_similarity,
    calculate_conceptual_similarity,
)

# --- Tool Input Schemas (using Pydantic) ---

class VisualSimilarityInput(BaseModel):
    """Input schema for the visual wordmark similarity tool."""
    applicant_wordmark: Wordmark = Field(..., description="Applicant's wordmark details.")
    opponent_wordmark: Wordmark = Field(..., description="Opponent's wordmark details.")

class WordmarkSimilarityInput(VisualSimilarityInput):
    """Input schema for wordmark similarity tools (visual, aural, conceptual)."""
    pass # Inherits fields from VisualSimilarityInput

class GoodsServicesSimilarityInput(BaseModel):
    """Input schema for the goods and services similarity tool."""
    applicant_goods_services: List[GoodsService] = Field(
        ..., description="List of applicant's goods/services."
    )
    opponent_goods_services: List[GoodsService] = Field(
        ..., description="List of opponent's goods/services (currently unused by underlying function but kept for potential future use)."
    )
    # Optional: Add threshold if you want the agent to control it
    # similarity_threshold: float = Field(default=0.8, description="Cosine similarity threshold.")


# --- Create Toolset ---
similarity_toolset = Toolset(
    name="trademark_similarity_tools",
    description="Tools for calculating various types of trademark similarity"
)

# --- ADK Tools ---

@similarity_toolset.tool(
    name="calculate_visual_wordmark_similarity",
    description="Calculates the normalized visual similarity between two wordmarks using Levenshtein ratio."
)
def calculate_visual_wordmark_similarity_tool(input_data: WordmarkSimilarityInput) -> float:
    """
    Calculates the normalized visual similarity between two wordmarks using Levenshtein ratio.

    Args:
        input_data: An object containing the applicant and opponent Wordmark objects.

    Returns:
        A float between 0.0 (dissimilar) and 1.0 (identical) representing visual similarity.
    """
    return calculate_visual_similarity(
        mark1=input_data.applicant_wordmark,
        mark2=input_data.opponent_wordmark
    )

@similarity_toolset.tool(
    name="calculate_goods_services_similarity",
    description="Calculates the aggregate semantic similarity between two lists of goods and services asynchronously."
)
async def calculate_goods_services_similarity_tool(input_data: GoodsServicesSimilarityInput) -> Optional[float]:
    """
    Calculates the aggregate semantic similarity between two lists of goods and services asynchronously.

    This tool generates embeddings for the applicant's terms and performs a vector
    search against indexed opponent/existing terms in the database to find the best matches.
    It returns an aggregated similarity score (0.0 to 1.0).

    Args:
        input_data: An object containing lists of applicant and opponent GoodsService objects.

    Returns:
        An aggregate similarity score (float between 0.0 and 1.0), or None if calculation fails.
    """
    # Note: The underlying function currently doesn't use opponent_goods_services directly,
    # relying on DB search instead. This input is kept for potential future flexibility.
    # Await the call to the async similarity function
    return await calculate_goods_services_similarity(
        applicant_gs=input_data.applicant_goods_services,
        opponent_gs=input_data.opponent_goods_services,
        # Pass threshold if included in input_data and needed by the function
        # similarity_threshold=input_data.similarity_threshold
    )

@similarity_toolset.tool(
    name="calculate_aural_wordmark_similarity",
    description="Calculates the normalized aural (sound) similarity between two wordmarks."
)
def calculate_aural_wordmark_similarity_tool(input_data: WordmarkSimilarityInput) -> float:
    """
    Calculates the normalized aural (sound) similarity between two wordmarks.

    Uses the Double Metaphone phonetic encoding and Levenshtein ratio on the codes.

    Args:
        input_data: An object containing the applicant and opponent Wordmark objects.

    Returns:
        A float between 0.0 (dissimilar) and 1.0 (identical) representing aural similarity.
    """
    return calculate_aural_similarity(
        mark1=input_data.applicant_wordmark,
        mark2=input_data.opponent_wordmark
    )

@similarity_toolset.tool(
    name="calculate_conceptual_wordmark_similarity",
    description="Calculates the conceptual similarity between two wordmarks using embeddings."
)
def calculate_conceptual_wordmark_similarity_tool(input_data: WordmarkSimilarityInput) -> Optional[float]:
    """
    Calculates the conceptual similarity between two wordmarks using embeddings.

    Generates embeddings for the wordmarks and returns their cosine similarity (scaled 0-1).

    Args:
        input_data: An object containing the applicant and opponent Wordmark objects.

    Returns:
        A float between 0.0 (dissimilar) and 1.0 (identical) representing conceptual similarity,
        or None if embedding generation fails.
    """
    return calculate_conceptual_similarity(
        mark1=input_data.applicant_wordmark,
        mark2=input_data.opponent_wordmark
    )

# TODO: Add tools for aural and conceptual similarity once implemented. 