"""
Functions for generating text embeddings using sentence-transformers.
"""

from typing import List, Optional

from sentence_transformers import SentenceTransformer


# Initialize the model globally for efficiency
# Consider making the model name configurable (e.g., via environment variable)
# 'all-MiniLM-L6-v2' is a good starting point (384 dimensions)
# 'all-mpnet-base-v2' is another strong model (768 dimensions)
# However, the DB schema expects 1536 dimensions. We need a model that outputs this.
# Let's use 'text-embedding-3-large' (OpenAI, needs separate handling) or find a suitable sentence-transformer.
# For now, using a placeholder model name that *might* not match 1536 dims for demonstration.
# IMPORTANT: Ensure the chosen model ACTUALLY outputs 1536 dimensions for the DB schema.
# Using 'all-mpnet-base-v2' (768 dims) as a placeholder - THIS NEEDS REPLACEMENT.
# MODEL_NAME = 'all-mpnet-base-v2' # <-- *** Placeholder - Needs model outputting 1536 dims ***
# Using all-MiniLM-L6-v2 as requested (384 dimensions)
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Loads and returns the sentence transformer model, caching it globally."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            # Log the error appropriately here
            print(f"Error loading SentenceTransformer model '{MODEL_NAME}': {e}")
            raise
    return _model


def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generates a vector embedding for the given text.

    Uses a pre-trained SentenceTransformer model.

    Args:
        text: The input text (e.g., a goods/services term) to embed.

    Returns:
        A list of floats representing the embedding, or None if embedding fails.
    """
    if not text: # Handle empty input
        return None

    try:
        model = _get_embedding_model()
        # The model's encode function returns a numpy array, convert it to list
        embedding = model.encode(text, convert_to_numpy=False)
        # Ensure the output is List[float] - encode might return ndarray or Tensor
        if hasattr(embedding, 'tolist'): # Handle numpy array
            return embedding.tolist()
        elif isinstance(embedding, list):
            return [float(val) for val in embedding] # Ensure elements are floats
        else:
            # Log an error or handle unexpected type
            print(f"Unexpected embedding type: {type(embedding)}")
            return None

    except Exception as e:
        # Log the error appropriately here
        print(f"Error generating embedding for text '{text[:50]}...': {e}")
        return None 