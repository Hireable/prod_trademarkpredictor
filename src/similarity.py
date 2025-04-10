# src/similarity.py
"""
Functions for calculating trademark similarity metrics.

Includes visual similarity based on Levenshtein distance and
goods/services similarity based on vector embeddings.
"""

import Levenshtein
import numpy as np
from typing import List, Optional

# Assuming db functions handle sessions appropriately or we manage them here
from src.db import find_similar_goods_services, get_async_session # Import async version
from src.embeddings import generate_embedding
from src.models import Wordmark, GoodsService

# Import for aural similarity
from metaphone import doublemetaphone

# Import for conceptual similarity (using numpy for cosine calculation)
import numpy as np


def calculate_visual_similarity(mark1: Wordmark, mark2: Wordmark) -> float:
    """
    Calculates the normalized visual similarity between two wordmarks.

    Uses the Levenshtein ratio algorithm, which measures the similarity
    between two strings as a float between 0.0 and 1.0 (inclusive).
    1.0 indicates identical strings, 0.0 indicates maximum dissimilarity.

    The comparison is case-insensitive. Empty strings are handled according
    to the Levenshtein library's behavior (ratio is 0.0 if one string is
    empty, 1.0 if both are empty).

    Args:
        mark1: The first Wordmark object.
        mark2: The second Wordmark object.

    Returns:
        A float between 0.0 and 1.0 representing the normalized
        visual similarity ratio.
    """
    text1: str = mark1.mark_text.lower() if mark1 and mark1.mark_text else ""
    text2: str = mark2.mark_text.lower() if mark2 and mark2.mark_text else ""

    # Levenshtein.ratio returns 1.0 for two empty strings, 0.0 if one is empty.
    # It handles None implicitly if converted to "" first.
    similarity_ratio: float = Levenshtein.ratio(text1, text2)

    return similarity_ratio


async def calculate_goods_services_similarity( # Make the function async
    applicant_gs: List[GoodsService],
    opponent_gs: List[GoodsService], # Currently unused, assumes opponent data is in DB
    similarity_threshold: float = 0.8 # Example threshold for cosine similarity (1 - distance)
) -> Optional[float]:
    """
    Calculates an aggregate similarity score between two lists of goods/services asynchronously.

    This version generates embeddings for the applicant's terms and searches
    for the most similar terms in the database (assumed to contain opponent/indexed data)
    using asynchronous database operations.
    It averages the similarity scores of the best matches found for each applicant term.

    Args:
        applicant_gs: List of GoodsService objects for the applicant.
        opponent_gs: List of GoodsService objects for the opponent (currently unused).
        similarity_threshold: Minimum cosine similarity score to consider a match.

    Returns:
        An aggregate similarity score (0.0 to 1.0), or None if calculation fails
        (e.g., no applicant terms, embedding errors, no matches found).
    """
    if not applicant_gs:
        return 0.0 # Or None, depending on desired handling of empty lists

    AsyncSessionLocal = await get_async_session() # Await the async session factory
    all_min_distances = [] # Store the minimum distance found for each applicant term

    # Use async session context manager
    async with AsyncSessionLocal() as session:
        # Optionally use session.begin() to manage transactions if storing embeddings here
        # async with session.begin():
        try:
            for app_item in applicant_gs:
                if not app_item.term:
                    continue

                # generate_embedding is assumed synchronous (CPU-bound is often fine)
                app_embedding = generate_embedding(app_item.term)
                if app_embedding is None:
                    print(f"Warning: Could not generate embedding for applicant term: {app_item.term}")
                    continue

                # Find the most similar term(s) asynchronously
                similar_items = await find_similar_goods_services(
                    app_embedding,
                    limit=1,
                    session=session # Pass the async session
                )

                if similar_items:
                    # similar_items is List[Tuple[GoodsServiceOrm, float]]
                    best_match_distance = similar_items[0][1]
                    all_min_distances.append(best_match_distance)
                else:
                    print(f"Warning: No similar item found in DB for applicant term: {app_item.term}")
                    all_min_distances.append(2.0) # Max cosine distance

            if not all_min_distances:
                print("Warning: No comparable goods/services found after embedding/search.")
                return None

            avg_min_distance = np.mean(all_min_distances)
            aggregate_similarity = max(0.0, 1.0 - (avg_min_distance / 2.0))

            return aggregate_similarity

        except Exception as e:
            print(f"Error calculating goods/services similarity: {e}")
            # Rollback is handled automatically if using session.begin()
            # If not using session.begin(), explicit rollback might be needed depending on error and operations
            # await session.rollback() # Consider if needed without session.begin()
            return None
        # No finally block needed to close session, context manager handles it


def calculate_aural_similarity(mark1: Wordmark, mark2: Wordmark) -> float:
    """
    Calculates the normalized aural similarity between two wordmarks.

    Uses the Double Metaphone algorithm to generate phonetic codes for the words
    and then calculates the Levenshtein ratio between these codes.
    Comparison is case-insensitive.

    Args:
        mark1: The first Wordmark object.
        mark2: The second Wordmark object.

    Returns:
        A float between 0.0 and 1.0 representing the normalized aural similarity ratio.
    """
    text1: str = mark1.mark_text.strip() if mark1 and mark1.mark_text else ""
    text2: str = mark2.mark_text.strip() if mark2 and mark2.mark_text else ""

    if not text1 and not text2:
        return 1.0 # Both empty
    if not text1 or not text2:
        return 0.0 # One empty

    # Generate Double Metaphone codes (primary, alternate)
    # We compare primary to primary and alternate to alternate if available,
    # taking the max similarity. Simpler approach: just use primary.
    codes1 = doublemetaphone(text1)
    codes2 = doublemetaphone(text2)

    primary_code1 = codes1[0]
    primary_code2 = codes2[0]

    # Calculate similarity based on primary codes
    similarity = Levenshtein.ratio(primary_code1, primary_code2)

    # Optional: Consider alternate codes for potentially higher similarity
    # alt_code1 = codes1[1]
    # alt_code2 = codes2[1]
    # if alt_code1 and alt_code2:
    #     alt_similarity = Levenshtein.ratio(alt_code1, alt_code2)
    #     similarity = max(similarity, alt_similarity)
    # Could also compare primary1 vs alt2 and primary2 vs alt1 if needed.

    return similarity


def calculate_conceptual_similarity(mark1: Wordmark, mark2: Wordmark) -> Optional[float]:
    """
    Calculates the conceptual similarity between two wordmarks based on embeddings.

    Generates embeddings for both wordmarks using the sentence transformer model
    and calculates the cosine similarity between them.

    Args:
        mark1: The first Wordmark object.
        mark2: The second Wordmark object.

    Returns:
        A float between 0.0 (dissimilar) and 1.0 (identical) representing cosine similarity,
        or None if embedding generation fails for either mark.
    """
    text1: str = mark1.mark_text.strip() if mark1 and mark1.mark_text else ""
    text2: str = mark2.mark_text.strip() if mark2 and mark2.mark_text else ""

    if not text1 or not text2:
        # Cannot compare if one is empty in this context
        return None # Or potentially 0.0 depending on requirements

    try:
        emb1 = generate_embedding(text1)
        emb2 = generate_embedding(text2)

        if emb1 is None or emb2 is None:
            print("Warning: Could not generate embedding for one or both wordmarks for conceptual similarity.")
            return None

        # Calculate Cosine Similarity
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Clamp the value between 0 and 1 (cosine similarity is -1 to 1)
        # We map it to 0-1 where 1 is most similar
        similarity_score = (cosine_sim + 1) / 2.0

        return max(0.0, min(1.0, similarity_score)) # Ensure bounds

    except Exception as e:
        print(f"Error calculating conceptual similarity: {e}")
        return None


# TODO: Refine conceptual similarity - simple embedding comparison might not capture legal nuances.
# TODO: Consider adding normalization or weighting to different similarity scores later.
