"""
FastAPI application for trademark similarity prediction.

This module provides HTTP endpoints for comparing trademarks and predicting
opposition outcomes using similarity analysis and LLM-powered reasoning.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from trademark_core import models
from trademark_core.similarity import (
    calculate_overall_similarity,
    calculate_goods_services_similarity
)
from trademark_core.llm import generate_prediction_reasoning

# Initialize FastAPI app
app = FastAPI(
    title="Trademark Similarity API",
    description="API for predicting trademark opposition outcomes",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

class MarkComparisonRequest(BaseModel):
    """Request model for mark comparison endpoint."""
    applicant: models.Mark
    opponent: models.Mark
    applicant_goods: List[models.GoodService]
    opponent_goods: List[models.GoodService]

class PredictionResponse(BaseModel):
    """Response model for trademark opposition prediction."""
    mark_comparison: models.MarkComparison
    goods_services_similarity: float
    likelihood_of_confusion: bool
    reasoning: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_opposition(request: MarkComparisonRequest) -> PredictionResponse:
    """
    Predict the outcome of a trademark opposition based on mark and goods/services comparison.
    
    Args:
        request: The comparison request containing applicant and opponent details
        
    Returns:
        PredictionResponse: Detailed prediction results and reasoning
    """
    # Calculate mark similarities
    mark_comparison = await calculate_overall_similarity(
        request.applicant,
        request.opponent
    )
    
    # Calculate goods/services similarity
    goods_similarity = calculate_goods_services_similarity(
        request.applicant_goods,
        request.opponent_goods
    )
    
    # Determine likelihood of confusion
    # High similarity in either marks or goods/services can lead to confusion
    mark_high_similarity = mark_comparison.overall in ["identical", "high"]
    goods_high_similarity = goods_similarity > 0.7
    
    likelihood_of_confusion = mark_high_similarity or (
        mark_comparison.overall == "moderate" and goods_high_similarity
    )
    
    # Generate reasoning using LLM
    reasoning = await generate_prediction_reasoning(
        mark_comparison=mark_comparison,
        goods_similarity=goods_similarity,
        likelihood_of_confusion=likelihood_of_confusion,
        request=request
    )
    
    return PredictionResponse(
        mark_comparison=mark_comparison,
        goods_services_similarity=goods_similarity,
        likelihood_of_confusion=likelihood_of_confusion,
        reasoning=reasoning
    ) 