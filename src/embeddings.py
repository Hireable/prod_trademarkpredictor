"""
Functions for generating text embeddings using sentence-transformers and specialized models.
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple, Union

from sentence_transformers import SentenceTransformer
# Add imports for LegalBERT
from transformers import AutoTokenizer, AutoModel
import torch

# Import the logger
from src.logger import get_logger, exception

# Initialize logger
logger = get_logger(__name__)

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
# Using all-MiniLM-L6-v2 for 384-dimension embeddings
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
_model: Optional[SentenceTransformer] = None
_model_lock = asyncio.Lock()  # Add a lock for thread safety

# LegalBERT model from Hugging Face
LEGAL_BERT_MODEL = 'nlpaueb/legal-bert-base-uncased'
_legal_bert_tokenizer: Optional[AutoTokenizer] = None
_legal_bert_model: Optional[AutoModel] = None
_legal_bert_lock = asyncio.Lock()  # Add a lock for thread safety


def _get_embedding_model() -> SentenceTransformer:
    """Loads and returns the sentence transformer model, caching it globally."""
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            _model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Embedding model loaded successfully: {MODEL_NAME}")
        except Exception as e:
            logger.exception(f"Error loading SentenceTransformer model", exc=e, model=MODEL_NAME)
            raise
    return _model


def _sync_generate_embedding(text: str) -> Optional[List[float]]:
    """
    Synchronous implementation of embedding generation - for use with asyncio.to_thread().
    
    Args:
        text: The input text to embed.
        
    Returns:
        A list of floats representing the embedding, or None if embedding fails.
    """
    if not text:  # Handle empty input
        return None

    try:
        model = _get_embedding_model()
        # The model's encode function returns a numpy array, convert it to list
        embedding = model.encode(text, convert_to_numpy=False)
        # Ensure the output is List[float]
        if hasattr(embedding, 'tolist'):  # Handle numpy array
            return embedding.tolist()
        elif isinstance(embedding, list):
            return [float(val) for val in embedding]  # Ensure elements are floats
        else:
            # Log an error for unexpected type
            logger.error(f"Unexpected embedding type", type=str(type(embedding)))
            return None

    except Exception as e:
        # Log the error with context
        exception(
            f"Error generating embedding for text", 
            exc=e, 
            text_snippet=text[:50] + ("..." if len(text) > 50 else "")
        )
        return None


async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Asynchronously generates a vector embedding for the given text.
    
    Uses asyncio.to_thread() to run the CPU-bound embedding generation
    in a separate thread, allowing other async tasks to run concurrently.
    
    Args:
        text: The input text (e.g., a goods/services term) to embed.
        
    Returns:
        A list of floats representing the embedding, or None if embedding fails.
    """
    if not text:  # Handle empty input
        return None
    
    # Use the lock to prevent multiple threads from initializing the model simultaneously
    async with _model_lock:
        # Ensure model is loaded (lazy loading)
        if _model is None:
            try:
                _model = _get_embedding_model()
            except Exception as e:
                logger.exception("Failed to load embedding model", exc=e)
                return None
    
    # Run the CPU-bound embedding generation in a separate thread
    try:
        # This effectively makes the synchronous operation non-blocking
        return await asyncio.to_thread(_sync_generate_embedding, text)
    except Exception as e:
        # Log any errors that occur during thread execution
        exception(
            "Error in async embedding generation", 
            exc=e, 
            text_snippet=text[:50] + ("..." if len(text) > 50 else "")
        )
        return None


def _get_legal_bert_model() -> Tuple[AutoTokenizer, AutoModel]:
    """Loads and returns the LegalBERT model and tokenizer, caching them globally."""
    global _legal_bert_tokenizer, _legal_bert_model
    if _legal_bert_tokenizer is None or _legal_bert_model is None:
        try:
            logger.info(f"Loading LegalBERT model: {LEGAL_BERT_MODEL}")
            _legal_bert_tokenizer = AutoTokenizer.from_pretrained(LEGAL_BERT_MODEL)
            _legal_bert_model = AutoModel.from_pretrained(LEGAL_BERT_MODEL)
            logger.info(f"LegalBERT model loaded successfully: {LEGAL_BERT_MODEL}")
        except Exception as e:
            logger.exception(f"Error loading LegalBERT model", exc=e, model=LEGAL_BERT_MODEL)
            raise
    return _legal_bert_tokenizer, _legal_bert_model


def _sync_generate_legal_embedding(text: str) -> Optional[List[float]]:
    """
    Synchronous implementation of LegalBERT embedding generation.
    
    Args:
        text: The input text to embed with LegalBERT.
        
    Returns:
        A list of floats representing the embedding, or None if embedding fails.
    """
    if not text:  # Handle empty input
        return None

    try:
        tokenizer, model = _get_legal_bert_model()
        
        # Tokenize and prepare for model
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding as sentence representation (common practice)
        sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Convert to list
        return sentence_embedding.tolist()

    except Exception as e:
        # Log the error with context
        exception(
            f"Error generating LegalBERT embedding for text", 
            exc=e, 
            text_snippet=text[:50] + ("..." if len(text) > 50 else "")
        )
        return None


async def generate_legal_embedding(text: str) -> Optional[List[float]]:
    """
    Asynchronously generates a LegalBERT embedding for the given text.
    
    Uses LegalBERT to generate embeddings that capture legal domain knowledge
    and legal terminology relationships better than general-purpose models.
    
    Args:
        text: The input text (e.g., a trademark term) to embed with legal context.
        
    Returns:
        A list of floats representing the legal-domain embedding, or None if embedding fails.
    """
    if not text:  # Handle empty input
        return None
    
    # Use the lock to prevent multiple threads from initializing the model simultaneously
    async with _legal_bert_lock:
        # Ensure model is loaded (lazy loading)
        if _legal_bert_tokenizer is None or _legal_bert_model is None:
            try:
                _legal_bert_tokenizer, _legal_bert_model = _get_legal_bert_model()
            except Exception as e:
                logger.exception("Failed to load LegalBERT model", exc=e)
                return None
    
    # Run the CPU-bound embedding generation in a separate thread
    try:
        # This effectively makes the synchronous operation non-blocking
        return await asyncio.to_thread(_sync_generate_legal_embedding, text)
    except Exception as e:
        # Log any errors that occur during thread execution
        exception(
            "Error in async LegalBERT embedding generation", 
            exc=e, 
            text_snippet=text[:50] + ("..." if len(text) > 50 else "")
        )
        return None


# Optional: Batch embedding generation
async def generate_embeddings_batch(
    texts: List[str], 
    batch_size: int = 32,
    use_legal_bert: bool = False
) -> Dict[str, Optional[List[float]]]:
    """
    Asynchronously generates embeddings for multiple texts in batches.
    
    This can be more efficient than generating embeddings one by one
    when processing large numbers of terms.
    
    Args:
        texts: List of input texts to embed.
        batch_size: Number of texts to process in each batch.
        use_legal_bert: Whether to use LegalBERT for legal-domain specific embeddings.
        
    Returns:
        A dictionary mapping each input text to its embedding vector,
        or None if embedding generation failed for that text.
    """
    # Initialize results dict
    results: Dict[str, Optional[List[float]]] = {text: None for text in texts}
    
    # Process in batches to avoid memory issues with large lists
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Create tasks for all texts in the batch
        if use_legal_bert:
            tasks = [generate_legal_embedding(text) for text in batch]
        else:
            tasks = [generate_embedding(text) for text in batch]
        
        # Wait for all embeddings in this batch
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, skipping any that raised exceptions
        for text, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.warning(f"Exception in batch embedding", 
                              exc_type=type(result).__name__,
                              text_snippet=text[:30])
                results[text] = None
            else:
                results[text] = result
    
    return results 