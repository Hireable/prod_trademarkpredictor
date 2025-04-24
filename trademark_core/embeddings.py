"""
Text embedding utilities powered by Sentence Transformers.

This module loads the all-MiniLM-L6-v2 model once and provides embedding functions
for trademark comparison.
"""

import asyncio
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Load the sentence transformer model once and cache it.
    
    Returns:
        SentenceTransformer: The loaded model
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


async def embed(text: str) -> List[float]:
    """
    Generate embeddings for input text using the MiniLM model.
    
    Args:
        text: The text to embed
        
    Returns:
        List[float]: The 384-dimensional embedding vector
    """
    # Run in a thread pool to avoid blocking async operations
    model = get_model()
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, lambda: model.encode(text))
    
    # Convert to Python list for JSON serialization
    return embedding.tolist() 