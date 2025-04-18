# tests/test_prediction_tools.py
"""Tests for trademark opposition prediction tools."""

import pytest
from typing import Dict, Optional

from src.models import (
    PredictionTaskInput,
    SimilarityScores,
    Trademark,
    Wordmark,
    GoodsService,
)
from src.tools.prediction_tools import (
    PredictionInput,
    predict_opposition_outcome_tool,
    _calculate_weighted_score,
)


# --- Test Data Setup ---

def create_test_trademark(identifier: str, mark_text: str, terms: Dict[str, int]) -> Trademark:
    """
    Helper function to create test trademark objects.
    
    Args:
        identifier: Trademark identifier
        mark_text: The text of the wordmark
        terms: Dict mapping term text to nice_class
        
    Returns:
        A Trademark object for testing
    """
    goods_services = [
        GoodsService(term=term, nice_class=nice_class)
        for term, nice_class in terms.items()
    ]
    
    return Trademark(
        identifier=identifier,
        wordmark=Wordmark(mark_text=mark_text),
        goods_services=goods_services
    )


def create_test_scores(
    visual: Optional[float] = None,
    aural: Optional[float] = None,
    conceptual: Optional[float] = None,
    goods_services: Optional[float] = None
) -> SimilarityScores:
    """
    Helper function to create test similarity scores.
    
    Args:
        visual: Visual similarity score (0.0-1.0)
        aural: Aural similarity score (0.0-1.0)
        conceptual: Conceptual similarity score (0.0-1.0)
        goods_services: Goods/services similarity score (0.0-1.0)
        
    Returns:
        A SimilarityScores object for testing
    """
    return SimilarityScores(
        visual_similarity=visual,
        aural_similarity=aural,
        conceptual_similarity=conceptual,
        goods_services_similarity=goods_services
    )


# --- Test Setup ---

# Test case for high similarity in all dimensions
HIGH_SIMILARITY_CASE = PredictionTaskInput(
    applicant_trademark=create_test_trademark(
        "APP123", "ACME", {"Software": 9, "Software development": 42}
    ),
    opponent_trademark=create_test_trademark(
        "OPP456", "ACMEE", {"Computer software": 9, "Programming services": 42}
    ),
    similarity_scores=create_test_scores(
        visual=0.8, aural=0.9, conceptual=0.8, goods_services=0.85
    )
)

# Test case for medium similarity
MEDIUM_SIMILARITY_CASE = PredictionTaskInput(
    applicant_trademark=create_test_trademark(
        "APP789", "BLUEFIN", {"Financial services": 36}
    ),
    opponent_trademark=create_test_trademark(
        "OPP012", "BLUEFOX", {"Banking services": 36}
    ),
    similarity_scores=create_test_scores(
        visual=0.6, aural=0.7, conceptual=0.5, goods_services=0.6
    )
)

# Test case for low similarity
LOW_SIMILARITY_CASE = PredictionTaskInput(
    applicant_trademark=create_test_trademark(
        "APP345", "MOUNTAINVIEW", {"Clothing": 25}
    ),
    opponent_trademark=create_test_trademark(
        "OPP678", "SEAVIEW", {"Food products": 30}
    ),
    similarity_scores=create_test_scores(
        visual=0.3, aural=0.2, conceptual=0.4, goods_services=0.1
    )
)

# Test case with missing scores
INCOMPLETE_SCORES_CASE = PredictionTaskInput(
    applicant_trademark=create_test_trademark(
        "APP901", "PARTIAL", {"Toys": 28}
    ),
    opponent_trademark=create_test_trademark(
        "OPP234", "PORTION", {"Games": 28}
    ),
    similarity_scores=create_test_scores(
        visual=0.5, aural=None, conceptual=0.6, goods_services=None
    )
)


# --- Tests ---

def test_weighted_score_calculation_complete_scores():
    """Test calculation of weighted scores with complete input."""
    scores = create_test_scores(
        visual=0.8, aural=0.6, conceptual=0.7, goods_services=0.9
    )
    
    weighted_score = _calculate_weighted_score(scores)
    
    # Expected: (0.8*0.3) + (0.6*0.3) + (0.7*0.1) + (0.9*0.3) = 0.76
    expected_score = 0.76
    assert weighted_score == pytest.approx(expected_score, abs=0.01)


def test_weighted_score_calculation_missing_scores():
    """Test calculation of weighted scores with some missing inputs."""
    scores = create_test_scores(
        visual=0.8, aural=None, conceptual=0.7, goods_services=0.9
    )
    
    weighted_score = _calculate_weighted_score(scores)
    
    # With aural missing, weights should be adjusted
    # Expected: (0.8*0.3) + (0.7*0.1) + (0.9*0.3) = 0.65, normalized by 0.7
    expected_score = 0.24 + 0.07 + 0.27
    expected_normalized = expected_score / 0.7
    assert weighted_score == pytest.approx(expected_normalized, abs=0.01)


def test_weighted_score_calculation_all_scores_missing():
    """Test calculation of weighted scores with all inputs missing."""
    scores = create_test_scores(
        visual=None, aural=None, conceptual=None, goods_services=None
    )
    
    weighted_score = _calculate_weighted_score(scores)
    
    # Should return None when no scores are available
    assert weighted_score is None


def test_prediction_high_similarity():
    """Test prediction with high similarity scores."""
    prediction_input = PredictionInput(prediction_task=HIGH_SIMILARITY_CASE)
    
    result = predict_opposition_outcome_tool(prediction_input)
    
    # Should predict successful opposition with high confidence
    assert "Succeed" in result.predicted_outcome
    assert result.confidence_score is not None
    assert result.confidence_score > 0.7
    assert result.reasoning is not None
    assert "likelihood of confusion" in result.reasoning.lower()


def test_prediction_medium_similarity():
    """Test prediction with medium similarity scores."""
    prediction_input = PredictionInput(prediction_task=MEDIUM_SIMILARITY_CASE)
    
    result = predict_opposition_outcome_tool(prediction_input)
    
    # Should predict partial success with medium confidence
    assert "Partially" in result.predicted_outcome
    assert result.confidence_score is not None
    assert 0.5 <= result.confidence_score <= 0.8
    assert result.reasoning is not None


def test_prediction_low_similarity():
    """Test prediction with low similarity scores."""
    prediction_input = PredictionInput(prediction_task=LOW_SIMILARITY_CASE)
    
    result = predict_opposition_outcome_tool(prediction_input)
    
    # Should predict unsuccessful opposition with medium-low confidence
    assert "Unlikely" in result.predicted_outcome
    assert result.confidence_score is not None
    assert result.confidence_score < 0.7
    assert result.reasoning is not None
    assert "differences" in result.reasoning.lower()


def test_prediction_missing_scores():
    """Test prediction with some missing similarity scores."""
    prediction_input = PredictionInput(prediction_task=INCOMPLETE_SCORES_CASE)
    
    result = predict_opposition_outcome_tool(prediction_input)
    
    # Should still produce a prediction
    assert result.predicted_outcome is not None
    assert result.confidence_score is not None
    assert result.reasoning is not None


def test_prediction_reasoning_content():
    """Test that prediction reasoning contains appropriate legal concepts."""
    prediction_input = PredictionInput(prediction_task=HIGH_SIMILARITY_CASE)
    
    result = predict_opposition_outcome_tool(prediction_input)
    
    # Should mention relevant legal principles
    assert "interdependence principle" in result.reasoning.lower()
    assert "similarity" in result.reasoning.lower()
    assert any(score_type in result.reasoning.lower() for score_type in 
              ["visual", "aural", "conceptual", "goods/services"])


def test_prediction_interdependence_principle():
    """Test that the prediction applies the interdependence principle correctly."""
    # Create a case with low visual but high goods/services similarity
    mixed_case = PredictionTaskInput(
        applicant_trademark=create_test_trademark(
            "APP555", "TOTALLY_DIFFERENT", {"Computer software": 9}
        ),
        opponent_trademark=create_test_trademark(
            "OPP555", "COMPLETELY_UNLIKE", {"Software programming": 9}
        ),
        similarity_scores=create_test_scores(
            visual=0.1, aural=0.2, conceptual=0.3, goods_services=0.9
        )
    )
    
    prediction_input = PredictionInput(prediction_task=mixed_case)
    result = predict_opposition_outcome_tool(prediction_input)
    
    # Even with low mark similarity, high goods/services similarity
    # should result in at least some possibility of confusion
    assert result.predicted_outcome != "Opposition Unlikely to Succeed" 