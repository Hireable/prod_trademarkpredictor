"""
Tests for the trademark_core.embeddings module.
"""

import pytest

from trademark_core import embeddings


@pytest.mark.asyncio
async def test_embed_dimensions():
    """Test that the embedding function returns a 384-dimension vector."""
    # Test with a sample trademark
    sample_text = "Apple Computers"
    embedding = await embeddings.embed(sample_text)
    
    # Check that it's a list of floats with 384 dimensions
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_embed_consistency():
    """Test that the embedding function is consistent for the same input."""
    # Test with a sample trademark
    sample_text = "Microsoft Windows"
    
    # Generate embeddings twice
    embedding1 = await embeddings.embed(sample_text)
    embedding2 = await embeddings.embed(sample_text)
    
    # Check that they are identical
    assert embedding1 == embedding2 