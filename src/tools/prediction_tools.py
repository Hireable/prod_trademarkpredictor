# src/tools/prediction_tools.py
"""
ADK Tools for trademark opposition prediction based on similarity scores.
"""

from typing import Optional, Dict, Tuple, List
from pydantic import BaseModel, Field

from src.models import (
    PredictionTaskInput, 
    PredictionResult, 
    SimilarityScores
)

# --- Tool Input Schemas ---

class PredictionInput(BaseModel):
    """Input schema for the trademark opposition prediction tool."""
    prediction_task: PredictionTaskInput = Field(
        ..., description="The prediction task input with trademark details and similarity scores."
    )

# --- Define Tool Functions ---

def predict_opposition_outcome_tool(input_data: PredictionInput) -> PredictionResult:
    """
    Predicts the likely outcome of a trademark opposition based on similarity scores.
    
    This tool analyzes various similarity dimensions (visual, aural, conceptual, 
    goods/services) to determine if an opposition is likely to succeed, partially 
    succeed, or fail. It provides reasoning based on UK/EU trademark law principles.
    
    Args:
        input_data: An object containing the trademarks being compared and their similarity scores.
        
    Returns:
        A PredictionResult containing the predicted outcome, confidence score, and reasoning.
    """
    # Extract prediction task data
    task = input_data.prediction_task
    scores = task.similarity_scores
    
    # Define threshold values
    HIGH_SIMILARITY = 0.70
    MEDIUM_SIMILARITY = 0.50
    LOW_SIMILARITY = 0.30
    
    # Track reasons for the prediction
    reasons: List[str] = []
    
    # Store similarity assessments
    assessments: Dict[str, str] = {}
    
    # Calculate an overall score (use weighted combination if available)
    # This uses the same logic as the overall_similarity_tool but allows for custom analysis
    overall_score = _calculate_weighted_score(scores)
    
    # Assess each similarity dimension
    if scores.visual_similarity is not None:
        if scores.visual_similarity >= HIGH_SIMILARITY:
            assessments["visual"] = "high"
            reasons.append(f"Visual similarity is high ({scores.visual_similarity:.2f}), as the marks share substantial visual elements.")
        elif scores.visual_similarity >= MEDIUM_SIMILARITY:
            assessments["visual"] = "medium"
            reasons.append(f"Visual similarity is moderate ({scores.visual_similarity:.2f}), with some shared visual elements.")
        else:
            assessments["visual"] = "low"
            reasons.append(f"Visual similarity is low ({scores.visual_similarity:.2f}), with few shared visual elements.")
    
    if scores.aural_similarity is not None:
        if scores.aural_similarity >= HIGH_SIMILARITY:
            assessments["aural"] = "high"
            reasons.append(f"Aural similarity is high ({scores.aural_similarity:.2f}), as the marks sound similar when spoken.")
        elif scores.aural_similarity >= MEDIUM_SIMILARITY:
            assessments["aural"] = "medium"
            reasons.append(f"Aural similarity is moderate ({scores.aural_similarity:.2f}), with some phonetic similarities.")
        else:
            assessments["aural"] = "low"
            reasons.append(f"Aural similarity is low ({scores.aural_similarity:.2f}), with different pronunciations.")
    
    if scores.conceptual_similarity is not None:
        if scores.conceptual_similarity >= HIGH_SIMILARITY:
            assessments["conceptual"] = "high"
            # Enhanced reasoning for conceptual similarity with legal domain knowledge
            reasons.append(f"Conceptual similarity is high ({scores.conceptual_similarity:.2f}), as the marks convey similar meanings or fall within the same conceptual category recognized in trademark law.")
            
            # Additional explanation for very high conceptual similarity
            if scores.conceptual_similarity >= 0.85:
                # Extract wordmarks for detailed reasoning
                app_mark = task.applicant_trademark.wordmark.mark_text
                opp_mark = task.opponent_trademark.wordmark.mark_text
                reasons.append(f"The marks '{app_mark}' and '{opp_mark}' share strong conceptual associations that would be recognized by the average consumer, increasing the likelihood of confusion.")
        elif scores.conceptual_similarity >= MEDIUM_SIMILARITY:
            assessments["conceptual"] = "medium"
            reasons.append(f"Conceptual similarity is moderate ({scores.conceptual_similarity:.2f}), with related but distinct meanings that may be associated by consumers.")
        else:
            assessments["conceptual"] = "low"
            reasons.append(f"Conceptual similarity is low ({scores.conceptual_similarity:.2f}), with different meanings or concepts.")
    
    if scores.goods_services_similarity is not None:
        if scores.goods_services_similarity >= HIGH_SIMILARITY:
            assessments["goods_services"] = "high"
            reasons.append(f"Goods/services similarity is high ({scores.goods_services_similarity:.2f}), indicating overlapping commercial scope.")
        elif scores.goods_services_similarity >= MEDIUM_SIMILARITY:
            assessments["goods_services"] = "medium"
            reasons.append(f"Goods/services similarity is moderate ({scores.goods_services_similarity:.2f}), with related commercial areas.")
        else:
            assessments["goods_services"] = "low"
            reasons.append(f"Goods/services similarity is low ({scores.goods_services_similarity:.2f}), with distinct commercial areas.")
    
    # Make prediction based on EU/UK trademark law principles:
    # 1. At least one similarity type must be high
    # 2. Goods/services must be at least moderately similar
    # 3. Apply the interdependence principle (lower similarity in one aspect can be offset by higher similarity in another)
    
    # Default prediction and confidence
    outcome = "Opposition Unlikely to Succeed"
    confidence = 0.5
    
    # Check for high similarity in any dimension
    high_similarities = [k for k, v in assessments.items() if v == "high"]
    medium_similarities = [k for k, v in assessments.items() if v == "medium"]
    
    # Basic opposition prediction logic
    gs_assessment = assessments.get("goods_services", "low")
    
    if gs_assessment == "high" and len(high_similarities) >= 1:
        # High similarity in goods/services and at least one mark similarity dimension
        outcome = "Opposition Likely to Succeed"
        confidence = min(0.9, 0.7 + (len(high_similarities) * 0.05))
        
        # Enhanced reasoning with legal principles
        legal_principle = "Following the interdependence principle in EU trademark law, the high similarity in goods/services combined with high similarity in mark characteristics creates a likelihood of confusion."
        
        # Add conceptual similarity specific reasoning if applicable
        if "conceptual" in high_similarities:
            legal_principle += " The conceptual similarity is particularly significant as established in cases like Lloyd Schuhfabrik Meyer (C-342/97), where the Court held that conceptual similarity can create a likelihood of confusion even with moderate visual or aural differences."
        
        reasons.append(legal_principle)
    
    elif gs_assessment in ["high", "medium"] and (len(high_similarities) >= 1 or len(medium_similarities) >= 2):
        # Moderate-to-high similarity in goods/services with substantial mark similarities
        outcome = "Opposition May Partially Succeed"
        confidence = 0.65 + (len(high_similarities) * 0.05)
        
        # Enhanced reasoning with legal principles
        legal_principle = "There is a moderate likelihood of confusion based on the similarities in specific aspects, applying the interdependence principle of EU trademark law."
        
        # Add conceptual similarity specific reasoning if applicable
        if "conceptual" in high_similarities or "conceptual" in medium_similarities:
            legal_principle += " As established in Canon Kabushiki Kaisha v Metro-Goldwyn-Mayer (C-39/97), the conceptual meaning can be an important element in the global assessment of likelihood of confusion."
        
        reasons.append(legal_principle)
    
    else:
        # Either low goods/services similarity or insufficient mark similarities
        confidence = 0.55 + (0.1 if gs_assessment == "low" else 0)
        
        # Enhanced reasoning with legal principles
        legal_principle = "The differences between the marks and/or the commercial areas are substantial enough that consumers are unlikely to be confused, applying EU trademark law principles."
        
        # Add conceptual dissimilarity specific reasoning if applicable
        if assessments.get("conceptual") == "low":
            legal_principle += " Following Sabel BV v Puma AG (C-251/95), where conceptual differences are clear, they can counteract visual and aural similarities, reducing the likelihood of confusion."
        
        reasons.append(legal_principle)
    
    # Add overall analysis
    if overall_score is not None:
        reasons.append(f"The calculated overall similarity score is {overall_score:.2f}, which supports this prediction.")
    
    # Combine reasons into a coherent reasoning paragraph
    reasoning = " ".join(reasons)
    
    # Return the prediction result
    return PredictionResult(
        predicted_outcome=outcome,
        confidence_score=confidence,
        reasoning=reasoning
    )

def _calculate_weighted_score(scores: SimilarityScores) -> Optional[float]:
    """
    Helper function to calculate a weighted overall similarity score.
    This is similar to the calculate_overall_similarity_tool but internal to this module.
    """
    # Define default weights
    weights: Dict[str, float] = {
        "visual_similarity": 0.30,
        "aural_similarity": 0.30,
        "conceptual_similarity": 0.10,
        "goods_services_similarity": 0.30,
    }
    
    # Collect available scores
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
        return None
    
    # Calculate weighted sum
    total_weight = 0.0
    weighted_sum = 0.0
    
    for score_name, score_value in available_scores:
        # Clamp individual scores
        score_value = max(0.0, min(1.0, score_value))
        weight = weights.get(score_name, 0.0)
        if weight > 0:
            weighted_sum += score_value * weight
            total_weight += weight
    
    if total_weight <= 0:
        # Simple average if no weights
        return sum(score for _, score in available_scores) / len(available_scores)
    
    # Normalize and return
    return weighted_sum / total_weight

# --- Create a List of Tool Functions ---
# ADK will wrap these functions into FunctionTools

prediction_tools = [
    predict_opposition_outcome_tool
] 