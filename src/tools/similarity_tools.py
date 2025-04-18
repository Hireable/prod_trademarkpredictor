# src/tools/similarity_tools.py
"""
ADK Tools for calculating various trademark similarity scores.
"""

from typing import List, Optional, Dict, Tuple

# Remove Toolset import, import FunctionTool instead
# from google.adk.tools.toolset import Toolset 
from pydantic import BaseModel, Field

from src.models import Wordmark, GoodsService, SimilarityScores # Added SimilarityScores import
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

class OverallSimilarityInput(BaseModel):
    """Input schema for the overall similarity calculation tool."""
    scores: SimilarityScores = Field(..., description="The individual similarity scores calculated previously.")
    # Optional: Weights could be passed in the input or configured elsewhere
    # weights: Optional[Dict[str, float]] = Field(default=None, description="Optional weights for each score type.")

# --- Define Tool Functions (keep existing functions) ---

# Note: The @tool decorator is removed. We'll wrap these functions manually.

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
    return await calculate_goods_services_similarity(
        applicant_gs=input_data.applicant_goods_services,
        opponent_gs=input_data.opponent_goods_services,
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

# Define default weights for overall similarity calculation
DEFAULT_WEIGHTS: Dict[str, float] = {
    "visual_similarity": 0.30,
    "aural_similarity": 0.30,
    "conceptual_similarity": 0.10,
    "goods_services_similarity": 0.30,
}

def calculate_overall_similarity_tool(input_data: OverallSimilarityInput) -> Optional[float]:
    """
    Calculates a weighted overall similarity score from individual scores.
    
    This tool applies weights to each individual similarity score (visual, aural, 
    conceptual, goods/services) and computes a weighted average. It handles missing
    scores by excluding them from the calculation and normalizing the remaining weights.
    
    Args:
        input_data: An object containing the SimilarityScores.
        
    Returns:
        A single float representing the weighted overall similarity score (0.0 to 1.0),
        or None if no valid scores are available.
    """
    scores = input_data.scores
    weights = DEFAULT_WEIGHTS  # Use default weights for MVP
    
    available_scores: List[Tuple[str, float]] = []
    if scores.visual_similarity is not None:
        available_scores.append(("visual_similarity", scores.visual_similarity))
    if scores.aural_similarity is not None:
        available_scores.append(("aural_similarity", scores.aural_similarity))
    if scores.conceptual_similarity is not None:
        available_scores.append(("conceptual_similarity", scores.conceptual_similarity))
    if scores.goods_services_similarity is not None:
        available_scores.append(("goods_services_similarity", scores.goods_services_similarity))
        
    if not available_scores:
        return None  # Cannot calculate if no scores are present
        
    total_weight: float = 0.0
    weighted_sum: float = 0.0
    
    for score_name, score_value in available_scores:
        # Clamp individual scores before weighting
        score_value = max(0.0, min(1.0, score_value))
        weight = weights.get(score_name, 0.0)
        if weight > 0:  # Only consider scores with positive weights
            weighted_sum += score_value * weight
            total_weight += weight
            
    if total_weight <= 0:  # Use <= 0 to handle potential floating point issues
        # Fallback: simple average of available (clamped) scores if total_weight is zero or negative
        if available_scores:
            clamped_scores = [max(0.0, min(1.0, s)) for _, s in available_scores]
            return sum(clamped_scores) / len(clamped_scores) if clamped_scores else None
        else:
            return None
            
    # Normalize by the sum of weights used
    overall_score = weighted_sum / total_weight
    
    # Ensure final score is within bounds
    return max(0.0, min(1.0, overall_score))

# --- Create a List of Tool Functions --- 
# ADK will wrap these functions into FunctionTools automatically

trademark_similarity_tools = [
    calculate_visual_wordmark_similarity_tool,
    calculate_goods_services_similarity_tool,
    calculate_aural_wordmark_similarity_tool,
    calculate_conceptual_wordmark_similarity_tool,
    calculate_overall_similarity_tool  # Added the new overall similarity tool
]