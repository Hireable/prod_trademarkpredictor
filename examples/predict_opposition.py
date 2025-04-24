#!/usr/bin/env python
"""
Example script for calling the Trademark Similarity API.

This script demonstrates how to call the API endpoint to predict
the outcome of a trademark opposition case.
"""

import json
import sys
import httpx
import asyncio
from typing import Dict, Any, List

API_URL = "http://localhost:8000/predict"

# Sample opposition case
SAMPLE_REQUEST = {
    "applicant": {
        "wordmark": "SKYWORD",
        "is_registered": False
    },
    "opponent": {
        "wordmark": "SKYWORKS",
        "is_registered": True,
        "registration_number": "EU12345678"
    },
    "applicant_goods": [
        {
            "term": "Computer software for content marketing",
            "nice_class": 9
        },
        {
            "term": "Content marketing services",
            "nice_class": 35
        }
    ],
    "opponent_goods": [
        {
            "term": "Semiconductors",
            "nice_class": 9
        },
        {
            "term": "Design of integrated circuits",
            "nice_class": 42
        }
    ]
}

async def predict_opposition(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the API to predict trademark opposition outcome.
    
    Args:
        request_data: The request payload with trademark information
        
    Returns:
        Dict[str, Any]: The API response with prediction results
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=request_data,
            timeout=30.0  # Longer timeout for LLM processing
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {}
            
        return response.json()

def display_results(results: Dict[str, Any]) -> None:
    """
    Display the prediction results in a readable format.
    
    Args:
        results: The API response with prediction results
    """
    if not results:
        return
        
    print("\n====== TRADEMARK OPPOSITION PREDICTION ======\n")
    
    # Mark comparison
    mark_comparison = results.get("mark_comparison", {})
    print("MARK COMPARISON:")
    print(f"- Visual Similarity: {mark_comparison.get('visual', 'N/A')}")
    print(f"- Aural Similarity: {mark_comparison.get('aural', 'N/A')}")
    print(f"- Conceptual Similarity: {mark_comparison.get('conceptual', 'N/A')}")
    print(f"- Overall Similarity: {mark_comparison.get('overall', 'N/A')}")
    
    # Goods/Services
    print(f"\nGOODS/SERVICES SIMILARITY: {results.get('goods_services_similarity', 'N/A'):.2f}")
    
    # Likelihood of confusion
    likelihood = results.get("likelihood_of_confusion", False)
    print(f"\nLIKELIHOOD OF CONFUSION: {'Yes' if likelihood else 'No'}")
    
    # Reasoning
    print("\nREASONING:")
    print(results.get("reasoning", "No reasoning provided"))

async def main(request_data: Dict[str, Any] = None) -> None:
    """
    Main function to run the example.
    
    Args:
        request_data: Optional custom request data
    """
    # Use sample data if none provided
    if request_data is None:
        request_data = SAMPLE_REQUEST
        
    print("Calling Trademark Similarity API...")
    print(f"Comparing '{request_data['applicant']['wordmark']}' vs '{request_data['opponent']['wordmark']}'")
    
    results = await predict_opposition(request_data)
    display_results(results)

if __name__ == "__main__":
    # Check if custom JSON file path is provided
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as f:
                custom_data = json.load(f)
            asyncio.run(main(custom_data))
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            sys.exit(1)
    else:
        # Use default sample data
        asyncio.run(main()) 