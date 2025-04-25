"""
Core similarity calculation functions for trademark comparison.

This module provides functions to calculate visual, aural, conceptual, and goods/services
similarity between trademarks, operating entirely in memory without database dependencies.
"""

from typing import List, Optional, Dict, Tuple
import Levenshtein
from metaphone import doublemetaphone

from trademark_core import models
from trademark_core.conceptual import calculate_conceptual_similarity as calculate_conceptual_similarity_impl
from trademark_core.llm import calculate_goods_services_similarity_llm

def calculate_visual_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate visual similarity between two trademarks using Levenshtein distance.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    # Clean and normalize marks
    mark1 = mark1.lower().strip()
    mark2 = mark2.lower().strip()
    
    # Handle empty strings
    if not mark1 and not mark2:
        return 1.0
    if not mark1 or not mark2:
        return 0.0
    
    # Calculate Levenshtein ratio
    return Levenshtein.ratio(mark1, mark2)

def calculate_aural_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate aural (phonetic) similarity between two trademarks using Double Metaphone.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    # Clean and normalize marks
    mark1 = mark1.lower().strip()
    mark2 = mark2.lower().strip()
    
    # Handle empty strings
    if not mark1 and not mark2:
        return 1.0
    if not mark1 or not mark2:
        return 0.0
    
    # Get Double Metaphone codes
    code1_primary, code1_alt = doublemetaphone(mark1)
    code2_primary, code2_alt = doublemetaphone(mark2)
    
    # Calculate similarities using primary and alternate codes
    primary_sim = Levenshtein.ratio(code1_primary, code2_primary)
    
    # If alternates exist, consider them too
    alt_sims = []
    if code1_alt and code2_alt:
        alt_sims.append(Levenshtein.ratio(code1_alt, code2_alt))
    if code1_primary and code2_alt:
        alt_sims.append(Levenshtein.ratio(code1_primary, code2_alt))
    if code1_alt and code2_primary:
        alt_sims.append(Levenshtein.ratio(code1_alt, code2_primary))
    
    # Return highest similarity found
    return max([primary_sim] + alt_sims) if alt_sims else primary_sim

async def calculate_conceptual_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate conceptual similarity between two trademarks using Gemini 2.5 Pro.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    return await calculate_conceptual_similarity_impl(mark1, mark2)

async def calculate_goods_services_similarity(applicant_gs: List[models.GoodService], 
                                     opponent_gs: List[models.GoodService]) -> float:
    """
    Calculate similarity between two lists of goods/services using LLM analysis.
    
    Args:
        applicant_gs: List of applicant's goods/services
        opponent_gs: List of opponent's goods/services
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    if not applicant_gs or not opponent_gs:
        return 0.0
    
    return await calculate_goods_services_similarity_llm(applicant_gs, opponent_gs)

async def calculate_overall_similarity(mark1: models.Mark, mark2: models.Mark) -> models.MarkComparison:
    """
    Calculate overall similarity between two trademarks across all dimensions.
    
    Args:
        mark1: First trademark
        mark2: Second trademark
        
    Returns:
        MarkComparison: Comparison results for all dimensions
    """
    # Calculate individual similarities
    visual_sim = calculate_visual_similarity(mark1.wordmark, mark2.wordmark)
    aural_sim = calculate_aural_similarity(mark1.wordmark, mark2.wordmark)
    conceptual_sim = await calculate_conceptual_similarity(mark1.wordmark, mark2.wordmark)
    
    # Map float scores to EnumStr values
    def score_to_enum(score: float) -> models.EnumStr:
        if score > 0.9:
            return "identical"
        elif score > 0.7:
            return "high"
        elif score > 0.5:
            return "moderate"
        elif score > 0.3:
            return "low"
        else:
            return "dissimilar"
    
    # Convert scores to enums
    visual = score_to_enum(visual_sim)
    aural = score_to_enum(aural_sim)
    conceptual = score_to_enum(conceptual_sim)
    
    # Calculate overall similarity with weights
    weights = {
        'visual': 0.40,
        'aural': 0.35,
        'conceptual': 0.25
    }
    
    overall_score = (
        weights['visual'] * visual_sim +
        weights['aural'] * aural_sim +
        weights['conceptual'] * conceptual_sim
    )
    
    overall = score_to_enum(overall_score)
    
    return models.MarkComparison(
        visual=visual,
        aural=aural,
        conceptual=conceptual,
        overall=overall
    ) 