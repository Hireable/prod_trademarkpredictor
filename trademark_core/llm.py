"""
LLM integration for trademark similarity prediction reasoning.

This module provides integration with Google's Vertex AI Gemini 2.5 Pro model
for generating detailed legal reasoning based on trademark similarity analyses.
"""

import os
import json
from typing import Dict, Any, List, Optional
import logging

# Updated imports for Google Generative AI SDK
from google import genai
from google.api_core.exceptions import GoogleAPIError
from google.genai import types

from trademark_core import models
from trademark_core.similarity import (
    calculate_visual_similarity,
    calculate_aural_similarity,
    calculate_conceptual_similarity,
    calculate_goods_services_similarity,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Generative AI client
use_vertexai = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

try:
    if use_vertexai:
        # Using Vertex AI with Application Default Credentials (ADC)
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI")
            
        logger.info("Configuring Google Generative AI with Vertex AI")
        client = genai.Client(
            vertexai=True,  # Required for Vertex AI
            project=project_id,
            location=location,
            http_options=types.HttpOptions(api_version='v1')  # Use stable API version
        )
        logger.info(f"Successfully initialized Vertex AI client in {location}")
    else:
        # Using direct API access with API Key
        logger.info("Configuring Google Generative AI with API Key")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found, falling back to Vertex AI")
            if not project_id:
                raise ValueError("Neither GOOGLE_API_KEY nor GOOGLE_CLOUD_PROJECT found. Cannot initialize client.")
                
            # Fallback to Vertex AI
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                http_options=types.HttpOptions(api_version='v1')
            )
            logger.info(f"Successfully initialized Vertex AI client in {location} (fallback)")
        else:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version='v1')
            )
            logger.info("Successfully initialized API Key client")
        
except Exception as e:
    logger.error(f"Failed to initialize Google Generative AI client: {str(e)}")
    raise

# Model configuration
DEFAULT_MODEL = "gemini-2.5-pro-preview-03-25"  # Model name is the same for both Vertex AI and direct API
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 40
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# Function declarations for the LLM
visual_similarity_func = types.FunctionDeclaration(
    name="calculate_visual_similarity",
    description="Calculate visual similarity between two wordmarks using Levenshtein distance",
    parameters={
        "type": "OBJECT",
        "properties": {
            "mark1": {"type": "STRING", "description": "First trademark text"},
            "mark2": {"type": "STRING", "description": "Second trademark text"}
        },
        "required": ["mark1", "mark2"]
    }
)

aural_similarity_func = types.FunctionDeclaration(
    name="calculate_aural_similarity", 
    description="Calculate aural (phonetic) similarity between two wordmarks using Double Metaphone",
    parameters={
        "type": "OBJECT",
        "properties": {
            "mark1": {"type": "STRING", "description": "First trademark text"},
            "mark2": {"type": "STRING", "description": "Second trademark text"}
        },
        "required": ["mark1", "mark2"]
    }
)

conceptual_similarity_func = types.FunctionDeclaration(
    name="calculate_conceptual_similarity",
    description="Calculate conceptual similarity between two wordmarks using semantic embeddings",
    parameters={
        "type": "OBJECT",
        "properties": {
            "mark1": {"type": "STRING", "description": "First trademark text"},
            "mark2": {"type": "STRING", "description": "Second trademark text"}
        },
        "required": ["mark1", "mark2"]
    }
)

goods_services_similarity_func = types.FunctionDeclaration(
    name="calculate_goods_services_similarity",
    description="Calculate similarity between goods/services terms",
    parameters={
        "type": "OBJECT",
        "properties": {
            "applicant_goods": {
                "type": "ARRAY",
                "description": "Applicant's goods/services",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "term": {"type": "STRING", "description": "Goods/services term"},
                        "nice_class": {"type": "INTEGER", "description": "Nice classification (1-45)"}
                    },
                    "required": ["term", "nice_class"]
                }
            },
            "opponent_goods": {
                "type": "ARRAY",
                "description": "Opponent's goods/services",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "term": {"type": "STRING", "description": "Goods/services term"},
                        "nice_class": {"type": "INTEGER", "description": "Nice classification (1-45)"}
                    },
                    "required": ["term", "nice_class"]
                }
            }
        },
        "required": ["applicant_goods", "opponent_goods"]
    }
)

# Function implementations for the LLM
async def _calculate_visual_similarity_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of calculate_visual_similarity for LLM function calling."""
    mark1 = args.get("mark1", "")
    mark2 = args.get("mark2", "")
    
    similarity = calculate_visual_similarity(mark1, mark2)
    return {"similarity": similarity}

async def _calculate_aural_similarity_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of calculate_aural_similarity for LLM function calling."""
    mark1 = args.get("mark1", "")
    mark2 = args.get("mark2", "")
    
    similarity = calculate_aural_similarity(mark1, mark2)
    return {"similarity": similarity}

async def _calculate_conceptual_similarity_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of calculate_conceptual_similarity for LLM function calling."""
    mark1 = args.get("mark1", "")
    mark2 = args.get("mark2", "")
    
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    return {"similarity": similarity}

async def _calculate_goods_services_similarity_impl(args: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of calculate_goods_services_similarity for LLM function calling."""
    # Convert dict to GoodService objects
    applicant_goods = [
        models.GoodService(term=item["term"], nice_class=item["nice_class"])
        for item in args.get("applicant_goods", [])
    ]
    
    opponent_goods = [
        models.GoodService(term=item["term"], nice_class=item["nice_class"])
        for item in args.get("opponent_goods", [])
    ]
    
    similarity = calculate_goods_services_similarity(applicant_goods, opponent_goods)
    return {"similarity": similarity}

# Function dispatcher
FUNCTION_MAP = {
    "calculate_visual_similarity": _calculate_visual_similarity_impl,
    "calculate_aural_similarity": _calculate_aural_similarity_impl,
    "calculate_conceptual_similarity": _calculate_conceptual_similarity_impl,
    "calculate_goods_services_similarity": _calculate_goods_services_similarity_impl
}

async def handle_function_call(model_response) -> List[Dict[str, Any]]:
    """Process function calls from the model and return the results."""
    function_responses = []
    
    # Check if there are any function calls in the response
    for candidate in model_response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                function_name = function_call.name
                function_args = function_call.args
                
                logger.info(f"Function call: {function_name} with args: {function_args}")
                
                # Call the appropriate function
                if function_name in FUNCTION_MAP:
                    try:
                        function_impl = FUNCTION_MAP[function_name]
                        result = await function_impl(function_args)
                        
                        function_responses.append({
                            "name": function_name,
                            "response": result
                        })
                    except Exception as e:
                        logger.error(f"Error executing function {function_name}: {str(e)}")
                        function_responses.append({
                            "name": function_name,
                            "error": str(e)
                        })
                else:
                    logger.warning(f"Unknown function: {function_name}")
                    function_responses.append({
                        "name": function_name,
                        "error": "Function not implemented"
                    })
    
    return function_responses

async def generate_prediction_reasoning(
    mark_comparison: models.MarkComparison,
    goods_similarity: float,
    likelihood_of_confusion: bool,
    request: Any
) -> str:
    """
    Generate detailed reasoning for the prediction using Vertex AI's Gemini 2.5 Pro.
    
    Args:
        mark_comparison: The mark comparison results
        goods_similarity: The goods/services similarity score
        likelihood_of_confusion: The predicted likelihood of confusion
        request: The original comparison request
        
    Returns:
        str: Detailed reasoning for the prediction
        
    Raises:
        GoogleAPIError: If there is an error connecting to the AI service
    """
    try:
        # Format the prompt for the LLM
        prompt = f"""
        Analyze the following trademark opposition case and provide detailed reasoning:
        
        Applicant's Mark: {request.applicant.wordmark}
        Opponent's Mark: {request.opponent.wordmark}
        
        Similarity Analysis:
        - Visual Similarity: {mark_comparison.visual}
        - Aural Similarity: {mark_comparison.aural}
        - Conceptual Similarity: {mark_comparison.conceptual}
        - Overall Mark Similarity: {mark_comparison.overall}
        
        Goods/Services Similarity: {goods_similarity}
        
        Applicant's Goods/Services:
        {", ".join([f"{g.term} (Class {g.nice_class})" for g in request.applicant_goods])}
        
        Opponent's Goods/Services:
        {", ".join([f"{g.term} (Class {g.nice_class})" for g in request.opponent_goods])}
        
        Likelihood of Confusion: {'Yes' if likelihood_of_confusion else 'No'}
        
        You are a trademark law expert. Based on the above analysis, provide a detailed legal reasoning for the likelihood of confusion determination. 
        Explain the legal principles involved and how the similarities or differences between the marks and goods/services contribute to your conclusion.
        Include references to any relevant trademark law concepts. Structure your response clearly with appropriate headings.
        """

        # Generate content using the client's models interface
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a trademark law expert providing detailed legal analysis of trademark opposition cases. Focus on likelihood of confusion analysis under UK/EU trademark law.",
                temperature=0.3,
                top_p=0.95,
                top_k=20,
                max_output_tokens=4000,
            )
        )
        
        # Extract text from response
        reasoning = response.text.strip()
        
        logger.info("Successfully generated prediction reasoning")
        return reasoning
        
    except GoogleAPIError as e:
        logger.error(f"Error generating prediction reasoning: {str(e)}")
        raise 