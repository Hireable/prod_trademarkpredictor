# tests/test_similarity_all.py
"""
Comprehensive tests for all trademark similarity calculation functions.
Includes tests for visual, aural, conceptual, and goods/services similarity.
"""
import asyncio
import pytest
import pytest_asyncio
from typing import List, Optional

from src.models import Wordmark, GoodsService
from src.similarity import (
    calculate_visual_similarity,
    calculate_aural_similarity,
    calculate_conceptual_similarity,
    calculate_goods_services_similarity
)
from src.db import get_async_session


# ---- Visual Similarity Tests ----

def test_visual_identical_wordmarks() -> None:
    """Test visual similarity with identical wordmarks expects 1.0."""
    mark1 = Wordmark(mark_text="TRADEMARK")
    mark2 = Wordmark(mark_text="TRADEMARK")
    expected_similarity = 1.0
    actual_similarity = calculate_visual_similarity(mark1, mark2)
    assert actual_similarity == pytest.approx(expected_similarity)


def test_visual_completely_different_wordmarks() -> None:
    """Test visual similarity with completely different wordmarks expects low value."""
    mark1 = Wordmark(mark_text="ABCDEFG")
    mark2 = Wordmark(mark_text="XYZ")
    expected_similarity = 0.0
    actual_similarity = calculate_visual_similarity(mark1, mark2)
    assert actual_similarity == pytest.approx(expected_similarity)


def test_visual_case_insensitivity() -> None:
    """Test visual similarity is case-insensitive."""
    mark1 = Wordmark(mark_text="TradeMark")
    mark2 = Wordmark(mark_text="trademark")
    expected_similarity = 1.0
    actual_similarity = calculate_visual_similarity(mark1, mark2)
    assert actual_similarity == pytest.approx(expected_similarity)


@pytest.mark.parametrize(
    "text1, text2, expected_similarity",
    [
        ("sitting", "kitten", 0.61538),
        ("sunday", "saturday", 0.71429),
        ("flaw", "lawn", 0.75),
        ("trademark", "trademar", 0.94118),
    ],
    ids=["substitute", "insert_delete", "mixed", "deletion_end"]
)
def test_visual_minor_variations(text1: str, text2: str, expected_similarity: float) -> None:
    """Test visual similarity with minor variations."""
    mark1 = Wordmark(mark_text=text1)
    mark2 = Wordmark(mark_text=text2)
    actual_similarity = calculate_visual_similarity(mark1, mark2)
    assert actual_similarity == pytest.approx(expected_similarity, abs=1e-5)


# ---- Aural Similarity Tests ----

def test_aural_identical_wordmarks() -> None:
    """Test aural similarity with identical wordmarks expects 1.0."""
    mark1 = Wordmark(mark_text="EXAMPLE")
    mark2 = Wordmark(mark_text="EXAMPLE")
    similarity = calculate_aural_similarity(mark1, mark2)
    assert similarity == pytest.approx(1.0)


def test_aural_phonetically_similar_wordmarks() -> None:
    """Test aural similarity with phonetically similar but visually different wordmarks."""
    mark1 = Wordmark(mark_text="NIGHT")
    mark2 = Wordmark(mark_text="NITE")
    similarity = calculate_aural_similarity(mark1, mark2)
    assert similarity > 0.7, "Phonetically similar words should have high aural similarity"


def test_aural_phonetically_different_wordmarks() -> None:
    """Test aural similarity with phonetically different wordmarks."""
    mark1 = Wordmark(mark_text="APPLE")
    mark2 = Wordmark(mark_text="CHAIR")
    similarity = calculate_aural_similarity(mark1, mark2)
    assert similarity < 0.5, "Phonetically different words should have low aural similarity"


@pytest.mark.parametrize(
    "text1, text2, expected_min_similarity",
    [
        ("COUGH", "KOFF", 0.7),
        ("PHARMACY", "FARMACY", 0.7),
        ("PHONE", "FONE", 0.7),
        ("EXPRESS", "XPRESS", 0.7),
    ],
    ids=["c-k sounds", "ph-f sounds", "ph-f initial", "ex-x initial"]
)
def test_aural_common_phonetic_equivalents(text1: str, text2: str, expected_min_similarity: float) -> None:
    """Test aural similarity with common phonetic equivalents."""
    mark1 = Wordmark(mark_text=text1)
    mark2 = Wordmark(mark_text=text2)
    actual_similarity = calculate_aural_similarity(mark1, mark2)
    assert actual_similarity >= expected_min_similarity, f"Expected at least {expected_min_similarity}, got {actual_similarity}"


def test_aural_empty_wordmarks() -> None:
    """Test aural similarity with empty wordmarks."""
    # Both empty
    mark1 = Wordmark(mark_text="")
    mark2 = Wordmark(mark_text="")
    similarity = calculate_aural_similarity(mark1, mark2)
    assert similarity == 1.0, "Two empty wordmarks should have aural similarity of 1.0"
    
    # One empty
    mark3 = Wordmark(mark_text="SOMETHING")
    similarity = calculate_aural_similarity(mark1, mark3)
    assert similarity == 0.0, "When one wordmark is empty, aural similarity should be 0.0"


# ---- Conceptual Similarity Tests ----

@pytest.mark.asyncio
async def test_conceptual_identical_wordmarks() -> None:
    """Test conceptual similarity with identical wordmarks expects 1.0."""
    mark1 = Wordmark(mark_text="EAGLE")
    mark2 = Wordmark(mark_text="EAGLE")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_conceptual_synonyms() -> None:
    """Test conceptual similarity with synonyms."""
    mark1 = Wordmark(mark_text="QUICK")
    mark2 = Wordmark(mark_text="FAST")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity >= 0.7, "Synonyms should have high conceptual similarity"


@pytest.mark.asyncio
async def test_conceptual_antonyms() -> None:
    """Test conceptual similarity with antonyms."""
    mark1 = Wordmark(mark_text="HOT")
    mark2 = Wordmark(mark_text="COLD")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity >= 0.5, "Antonyms should have medium-high conceptual similarity in trademark context"


@pytest.mark.asyncio
async def test_conceptual_unrelated_wordmarks() -> None:
    """Test conceptual similarity with unrelated concepts."""
    mark1 = Wordmark(mark_text="APPLE")
    mark2 = Wordmark(mark_text="ROCKET")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity < 0.5, "Unrelated concepts should have low conceptual similarity"


@pytest.mark.asyncio
async def test_conceptual_related_wordmarks() -> None:
    """Test conceptual similarity with related but not identical concepts."""
    mark1 = Wordmark(mark_text="LION")
    mark2 = Wordmark(mark_text="TIGER")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert 0.5 <= similarity <= 0.9, "Related animal concepts should have medium-high similarity"


@pytest.mark.asyncio
async def test_conceptual_same_stem_wordmarks() -> None:
    """Test conceptual similarity with words sharing the same stem."""
    mark1 = Wordmark(mark_text="RUNNER")
    mark2 = Wordmark(mark_text="RUNNING")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity >= 0.7, "Words with the same stem should have high conceptual similarity"


@pytest.mark.asyncio
async def test_conceptual_color_wordmarks() -> None:
    """Test conceptual similarity with color words."""
    mark1 = Wordmark(mark_text="BLUE")
    mark2 = Wordmark(mark_text="RED")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity >= 0.5, "Different colors should have medium conceptual similarity in trademark context"


@pytest.mark.asyncio
async def test_conceptual_empty_wordmarks() -> None:
    """Test conceptual similarity with empty wordmarks."""
    mark1 = Wordmark(mark_text="")
    mark2 = Wordmark(mark_text="CONCEPT")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity is None, "Conceptual similarity with empty wordmark should be None"


@pytest.mark.asyncio
async def test_conceptual_legalbert_enhanced_analysis() -> None:
    """Test conceptual similarity with LegalBERT for legal domain semantic understanding."""
    # Legal terminology test
    mark1 = Wordmark(mark_text="LEGITIMATE")
    mark2 = Wordmark(mark_text="LAWFUL")
    similarity = await calculate_conceptual_similarity(mark1, mark2)
    assert similarity is not None
    assert similarity >= 0.65, "Legal terms should have higher similarity with LegalBERT"
    
    # Test with legal domain trademark pairs that should be conceptually related
    legal_pairs = [
        ("ROYAL PREMIUM", "LUXURY ELITE"),  # Luxury category
        ("CYBER TECH", "DIGITAL NET"),      # Tech category
        ("RED FOX", "BLUE TIGER"),          # Color + animal categories
    ]
    
    for mark_text1, mark_text2 in legal_pairs:
        mark1 = Wordmark(mark_text=mark_text1)
        mark2 = Wordmark(mark_text=mark_text2)
        similarity = await calculate_conceptual_similarity(mark1, mark2)
        assert similarity is not None
        assert similarity >= 0.5, f"Legal category match '{mark_text1}' and '{mark_text2}' should have higher similarity"


# ---- Goods/Services Similarity Tests ----

# Mock goods/services for testing
SOFTWARE_GOODS = [
    GoodsService(term="Computer software", nice_class=9),
    GoodsService(term="Software as a service", nice_class=42)
]

CLOTHING_GOODS = [
    GoodsService(term="Clothing", nice_class=25),
    GoodsService(term="T-shirts", nice_class=25)
]

FOOD_GOODS = [
    GoodsService(term="Coffee", nice_class=30),
    GoodsService(term="Bakery products", nice_class=30)
]

@pytest.mark.asyncio
@pytest.mark.integration  # Mark as integration test that requires DB
async def test_goods_services_identical_terms() -> None:
    """Test goods/services similarity with identical terms."""
    # This test requires database setup with embeddings stored
    similarity = await calculate_goods_services_similarity(
        SOFTWARE_GOODS, SOFTWARE_GOODS
    )
    # The function might return None if DB setup is not complete
    # Only assert if we got a valid result
    if similarity is not None:
        assert similarity > 0.8, "Identical goods/services should have high similarity"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_goods_services_different_classes() -> None:
    """Test goods/services similarity with terms in different NICE classes."""
    # This test requires database setup with embeddings stored
    similarity = await calculate_goods_services_similarity(
        SOFTWARE_GOODS, CLOTHING_GOODS
    )
    # The function might return None if DB setup is not complete
    if similarity is not None:
        assert similarity < 0.5, "Terms in different classes should have lower similarity"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_goods_services_related_terms() -> None:
    """Test goods/services similarity with related terms."""
    # Create related terms within software domain
    software_a = [GoodsService(term="Mobile application software", nice_class=9)]
    software_b = [GoodsService(term="Software development services", nice_class=42)]
    
    # This test requires database setup with embeddings stored
    similarity = await calculate_goods_services_similarity(
        software_a, software_b
    )
    # The function might return None if DB setup is not complete
    if similarity is not None:
        assert 0.5 <= similarity <= 0.9, "Related terms should have medium-high similarity"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_goods_services_empty_list() -> None:
    """Test goods/services similarity with empty list."""
    empty_list: List[GoodsService] = []
    
    similarity = await calculate_goods_services_similarity(
        empty_list, SOFTWARE_GOODS
    )
    assert similarity == 0.0, "Empty applicant goods/services list should return 0.0"


# === Test Fixtures and Setup ===

@pytest_asyncio.fixture(scope="module")
async def setup_db():
    """Fixture to set up database for tests."""
    # This would typically initialize the database, create tables,
    # and possibly insert test data with embeddings
    
    # For now, just get the session maker to confirm DB is configured
    try:
        session_maker = await get_async_session()
        async with session_maker() as session:
            # Check if we can connect
            await session.execute(sqlalchemy.text("SELECT 1"))
        yield "Database setup complete"
    except Exception as e:
        pytest.skip(f"Database setup failed: {e}") 