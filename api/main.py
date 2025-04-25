"""
FastAPI application for trademark similarity prediction.

This module provides HTTP endpoints for comparing trademarks and predicting
opposition outcomes using LLM-powered similarity analysis and reasoning.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from trademark_core import models
from trademark_core.llm import generate_full_prediction

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

@app.post("/predict", response_model=models.CasePrediction)
async def predict_opposition(request: models.PredictionRequest) -> models.CasePrediction:
    """
    Predict the outcome of a trademark opposition based on mark and goods/services comparison.
    
    Args:
        request: The prediction request containing applicant and opponent details
        
    Returns:
        CasePrediction: Detailed prediction results with mark comparison and outcome
    """
    try:
        # Use the LLM-centric approach to generate the full prediction
        prediction = await generate_full_prediction(request)
        return prediction
    except Exception as e:
        # Handle any errors from the LLM process
        raise HTTPException(
            status_code=500,
            detail=f"Error generating prediction: {str(e)}"
        ) 