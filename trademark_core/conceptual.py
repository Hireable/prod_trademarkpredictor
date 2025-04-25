"""
Conceptual similarity calculations for trademarks using LLM.

This module provides LLM-based conceptual similarity analysis using Gemini 2.5 Pro.
"""

import logging
from typing import Optional
from google import genai
from google.api_core.exceptions import GoogleAPIError
from google.genai import types

# Configure logging
logger = logging.getLogger(__name__)

async def calculate_conceptual_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate conceptual similarity between two trademarks using Gemini 2.5 Pro.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    try:
        return await calculate_conceptual_similarity_llm(mark1, mark2)
    except Exception as e:
        logger.error(f"LLM-based conceptual similarity failed: {str(e)}")
        # Return a conservative estimate when LLM fails
        return 0.5

async def calculate_conceptual_similarity_llm(mark1: str, mark2: str) -> float:
    """
    Calculate conceptual similarity between two trademarks using Gemini 2.5 Pro.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
        
    Raises:
        GoogleAPIError: If there is an error connecting to the AI service
    """
    from trademark_core.llm import client  # Import here to avoid circular dependency
    
    try:
        prompt = f"""
        As a trademark law expert, analyze the conceptual similarity between these two trademarks:
        Mark 1: {mark1}
        Mark 2: {mark2}
        
        Consider:
        1. The meaning, idea, or concept behind each mark
        2. Whether they share common roots, abbreviations, or conceptual elements
        3. Whether one mark is a shortened version or variant of the other
        4. The overall commercial impression they create
        
        First, explain your analysis.
        Then, on a new line starting with "SCORE:", provide a single decimal number between 0.0 (completely dissimilar concepts) and 1.0 (identical concepts).
        
        Example scores:
        - 1.0: Identical concepts (e.g., "CLOUD" vs "CLOUD", or "TECH" vs "TECHNOLOGY")
        - 0.8-0.9: Nearly identical concepts with minor variation
        - 0.5-0.7: Related but distinct concepts
        - 0.2-0.4: Weak conceptual connection
        - 0.0-0.1: No conceptual similarity
        """

        # Generate content using the client's models interface
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",  # Use latest model
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent scoring
                top_p=0.95,
                top_k=20,
                max_output_tokens=1000,
            )
        )
        
        # Extract the score from the response
        text = response.text.strip()
        
        # Find the score line
        for line in text.split('\n'):
            if line.startswith('SCORE:'):
                try:
                    score = float(line.replace('SCORE:', '').strip())
                    # Ensure score is within bounds
                    return max(0.0, min(1.0, score))
                except ValueError:
                    logger.error("Failed to parse similarity score from LLM response")
                    return 0.5  # Return conservative estimate on parsing failure
        
        # If no score found, return conservative estimate
        logger.warning("No similarity score found in LLM response")
        return 0.5
        
    except GoogleAPIError as e:
        logger.error(f"Error calculating conceptual similarity with LLM: {str(e)}")
        raise 