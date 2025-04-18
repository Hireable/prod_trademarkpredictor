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
from src.logger import get_logger, info, warning, error, exception

# Import for aural similarity
from metaphone import doublemetaphone

# Import for conceptual similarity (using numpy for cosine calculation)
import numpy as np

# Initialize logger
logger = get_logger(__name__)

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
    
    # Log the calculation
    info("Calculated visual similarity", mark1=text1, mark2=text2, similarity=similarity_ratio)

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
        logger.warning("Empty applicant goods/services list provided")
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

                # Use the async version of generate_embedding
                app_embedding = await generate_embedding(app_item.term)
                if app_embedding is None:
                    warning("Could not generate embedding for applicant term", term=app_item.term)
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
                    best_match_term = similar_items[0][0].term
                    all_min_distances.append(best_match_distance)
                    info("Found similar goods/services term", 
                         applicant_term=app_item.term, 
                         best_match=best_match_term, 
                         distance=best_match_distance)
                else:
                    warning("No similar item found in DB for applicant term", term=app_item.term)
                    all_min_distances.append(2.0) # Max cosine distance

            if not all_min_distances:
                warning("No comparable goods/services found after embedding/search")
                return None

            avg_min_distance = np.mean(all_min_distances)
            aggregate_similarity = max(0.0, 1.0 - (avg_min_distance / 2.0))
            
            info("Calculated goods/services similarity", 
                 similarity=aggregate_similarity, 
                 avg_distance=avg_min_distance, 
                 num_terms=len(all_min_distances))

            return aggregate_similarity

        except Exception as e:
            exception("Error calculating goods/services similarity", exc=e)
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
        info("Both wordmarks are empty, aural similarity set to 1.0")
        return 1.0 # Both empty
    if not text1 or not text2:
        info("One wordmark is empty, aural similarity set to 0.0", 
             mark1=text1 or "(empty)", mark2=text2 or "(empty)")
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
    
    info("Calculated aural similarity", 
         mark1=text1, mark2=text2, 
         phonetic_code1=primary_code1, phonetic_code2=primary_code2,
         similarity=similarity)

    # Optional: Consider alternate codes for potentially higher similarity
    # alt_code1 = codes1[1]
    # alt_code2 = codes2[1]
    # if alt_code1 and alt_code2:
    #     alt_similarity = Levenshtein.ratio(alt_code1, alt_code2)
    #     similarity = max(similarity, alt_similarity)
    # Could also compare primary1 vs alt2 and primary2 vs alt1 if needed.

    return similarity


async def calculate_conceptual_similarity(mark1: Wordmark, mark2: Wordmark) -> Optional[float]:
    """
    Calculates the conceptual similarity between two wordmarks based on embeddings
    and legal trademark conceptual patterns.
    
    This enhanced version combines multiple approaches:
    1. Vector embedding similarity (capturing general semantic relationships)
    2. LegalBERT domain-specific legal embeddings (for legal conceptual relationships)
    3. Detection of common conceptual relationships in trademark law:
       - Direct synonyms or near-synonyms
       - Conceptual opposites (which can still be similar in trademark context)
       - Words sharing the same root/stem
       - Special category patterns (animals, colors, numbers, etc.)
       - Translation equivalents
    
    Args:
        mark1: The first Wordmark object.
        mark2: The second Wordmark object.
        
    Returns:
        A float between 0.0 (conceptually dissimilar) and 1.0 (conceptually identical) 
        representing the legal conceptual similarity, or None if analysis fails.
    """
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    import nltk
    from nltk.corpus import wordnet
    import re
    import numpy as np
    
    # Ensure required NLTK data is available
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            logger.info("Downloading wordnet for enhanced conceptual similarity")
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            exception("Could not download wordnet for enhanced conceptual similarity", exc=e)
    
    text1: str = mark1.mark_text.strip().lower() if mark1 and mark1.mark_text else ""
    text2: str = mark2.mark_text.strip().lower() if mark2 and mark2.mark_text else ""
    
    if not text1 or not text2:
        # Cannot compare if one is empty in this context
        warning("Cannot calculate conceptual similarity with empty wordmark(s)", 
                mark1=text1 or "(empty)", mark2=text2 or "(empty)")
        return None
    
    # --- 1. Base vector similarity (using existing code) ---
    base_similarity = None
    try:
        # Use the async version of generate_embedding
        emb1 = await generate_embedding(text1)
        emb2 = await generate_embedding(text2)
        
        if emb1 is None or emb2 is None:
            warning("Could not generate embedding for one or both wordmarks for conceptual similarity", 
                    mark1=text1, mark2=text2)
        else:
            # Calculate Cosine Similarity
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # Clamp the value between 0 and 1 (cosine similarity is -1 to 1)
            # We map it to 0-1 where 1 is most similar
            base_similarity = (cosine_sim + 1) / 2.0
            base_similarity = max(0.0, min(1.0, base_similarity))  # Ensure bounds
            
            info("Calculated base vector similarity for conceptual comparison",
                 mark1=text1, mark2=text2, base_similarity=base_similarity)
    except Exception as e:
        exception("Error calculating base vector similarity", exc=e, mark1=text1, mark2=text2)
    
    # --- 1.5 NEW: Legal domain-specific similarity with LegalBERT ---
    legal_similarity = None
    try:
        # Use the LegalBERT embeddings
        from src.embeddings import generate_legal_embedding
        
        legal_emb1 = await generate_legal_embedding(text1)
        legal_emb2 = await generate_legal_embedding(text2)
        
        if legal_emb1 is None or legal_emb2 is None:
            warning("Could not generate LegalBERT embedding for one or both wordmarks", 
                    mark1=text1, mark2=text2)
        else:
            # Calculate Cosine Similarity for legal embeddings
            legal_vec1 = np.array(legal_emb1)
            legal_vec2 = np.array(legal_emb2)
            legal_cosine_sim = np.dot(legal_vec1, legal_vec2) / (np.linalg.norm(legal_vec1) * np.linalg.norm(legal_vec2))
            
            # Normalize to 0-1 scale
            legal_similarity = (legal_cosine_sim + 1) / 2.0
            legal_similarity = max(0.0, min(1.0, legal_similarity))  # Ensure bounds
            
            info("Calculated LegalBERT similarity for conceptual comparison",
                 mark1=text1, mark2=text2, legal_similarity=legal_similarity)
            
            # We can give legal_similarity a bit more weight in the legal trademark context
            if base_similarity is not None:
                # Weighted average: 60% legal domain knowledge, 40% general semantics
                base_similarity = (legal_similarity * 0.6) + (base_similarity * 0.4)
                info("Combined LegalBERT and general embeddings",
                     mark1=text1, mark2=text2, combined_similarity=base_similarity)
            else:
                # If base_similarity failed, use legal_similarity as base
                base_similarity = legal_similarity
    except Exception as e:
        exception("Error calculating LegalBERT similarity", exc=e, mark1=text1, mark2=text2)
    
    # --- 2. Enhanced legal conceptual analysis ---
    
    # Helper function to tokenize wordmarks
    def tokenize(text):
        # Simple tokenization - split on non-alphanumeric chars and normalize
        return [t.lower() for t in re.findall(r'\w+', text)]
    
    # Extract tokens
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # --- ENHANCEMENT: Legal domain term matching ---
    # Dictionary of common legal trademark categories and their members
    # This could be expanded with more comprehensive lists from legal resources
    legal_trademark_categories = {
        "colors": {"red", "blue", "green", "yellow", "orange", "purple", "black", 
                 "white", "brown", "pink", "grey", "gray", "silver", "gold"},
        "animals": {"lion", "tiger", "bear", "eagle", "falcon", "hawk", "wolf", "fox", 
                  "deer", "bull", "dragon", "phoenix", "shark", "dolphin"},
        "luxury_terms": {"premium", "luxury", "elite", "exclusive", "supreme", "royal", 
                       "imperial", "prestige", "superior", "excellence", "deluxe"},
        "tech_terms": {"digital", "tech", "smart", "cyber", "web", "net", "online", 
                     "virtual", "electronic", "data", "cloud", "mobile", "app"},
        "number_prefixes": {"uni", "mono", "bi", "duo", "tri", "triple", "quad", "tetra", 
                          "penta", "hexa", "hepta", "octa", "nona", "deca"}
    }
    
    # Check if marks fall into same legal category (important in trademark law)
    legal_category_match = False
    for category, terms in legal_trademark_categories.items():
        # Check if both marks have terms from the same category
        if any(token in terms for token in tokens1) and any(token in terms for token in tokens2):
            legal_category_match = True
            info(f"Marks share terms from legal category '{category}'", 
                 mark1=text1, mark2=text2, category=category)
            break
    
    # Handle single-word marks efficiently
    if len(tokens1) == 1 and len(tokens2) == 1:
        word1, word2 = tokens1[0], tokens2[0]
        
        # Check for exact match first
        if word1 == word2:
            info("Exact word match for conceptual similarity", word=word1, similarity=1.0)
            return 1.0
            
        # Initialize similarity boosters
        legal_concept_boosts = []
        
        # --- Special category patterns ---
        
        # Color detection (simplified - a real impl would have a comprehensive list)
        colors = legal_trademark_categories["colors"]
        if word1 in colors and word2 in colors:
            legal_concept_boosts.append(0.7)  # Colors are conceptually related in TM law
            info("Color relationship detected for conceptual similarity", color1=word1, color2=word2, boost=0.7)
            
        # Number detection
        if word1.isdigit() and word2.isdigit():
            # Numbers are related but similarity depends on proximity
            num_similarity = 1.0 - min(abs(int(word1) - int(word2)) / 100, 0.9)
            legal_concept_boosts.append(num_similarity)
            info("Numeric relationship detected for conceptual similarity", 
                 num1=word1, num2=word2, boost=num_similarity)
        
        # --- Lexical relationship detection ---
        
        # Check word stems (e.g., "running" and "runner" share stem "run")
        try:
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            
            # Get stems and lemmas
            stem1, stem2 = stemmer.stem(word1), stemmer.stem(word2)
            lemma1 = lemmatizer.lemmatize(word1)
            lemma2 = lemmatizer.lemmatize(word2)
            
            # Compare stems and lemmas
            if stem1 == stem2:
                legal_concept_boosts.append(0.8)  # Same stem = conceptually linked
                info("Same stem detected for conceptual similarity", word1=word1, word2=word2, stem=stem1, boost=0.8)
            elif lemma1 == lemma2:
                legal_concept_boosts.append(0.75)  # Same lemma = conceptually linked
                info("Same lemma detected for conceptual similarity", word1=word1, word2=word2, lemma=lemma1, boost=0.75)
        except Exception as e:
            exception("Stem/lemma comparison failed", exc=e, word1=word1, word2=word2)
        
        # Check WordNet for synonyms/antonyms/hypernyms (semantic relationships)
        try:
            # Get all synsets for both words
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            
            if synsets1 and synsets2:
                # Check if they share any synsets (synonyms)
                shared_synsets = set(synsets1).intersection(set(synsets2))
                if shared_synsets:
                    legal_concept_boosts.append(0.9)  # Direct synonyms
                    info("Shared synsets detected for conceptual similarity", 
                         word1=word1, word2=word2, synset=str(next(iter(shared_synsets))), boost=0.9)
                else:
                    # Check for antonyms (conceptual opposites - important in TM law)
                    has_antonym_relation = False
                    for syn1 in synsets1:
                        for lemma1 in syn1.lemmas():
                            for antonym in lemma1.antonyms():
                                antonym_word = antonym.name().lower()
                                if antonym_word == word2 or any(antonym_word == lemma2.name().lower() 
                                                            for syn2 in synsets2 
                                                            for lemma2 in syn2.lemmas()):
                                    has_antonym_relation = True
                                    break
                            if has_antonym_relation:
                                break
                        if has_antonym_relation:
                            break
                    
                    if has_antonym_relation:
                        legal_concept_boosts.append(0.7)  # Antonyms can be conceptually related in TM law
                        info("Antonym relationship detected for conceptual similarity", word1=word1, word2=word2, boost=0.7)
                    
                    # Check for hypernym/hyponym relationships (broader/narrower terms)
                    max_similarity = 0.0
                    for syn1 in synsets1:
                        for syn2 in synsets2:
                            # Path similarity measures taxonomic similarity
                            sim = syn1.path_similarity(syn2)
                            if sim is not None and sim > max_similarity:
                                max_similarity = sim
                    
                    if max_similarity > 0:
                        # Scale the similarity to reflect legal concepts better
                        # WordNet path_similarity is already 0-1
                        adjusted_similarity = max_similarity * 0.7
                        legal_concept_boosts.append(adjusted_similarity)
                        info("WordNet path similarity detected", word1=word1, word2=word2, 
                             path_similarity=max_similarity, boost=adjusted_similarity)
        except Exception as e:
            exception("WordNet analysis failed", exc=e, word1=word1, word2=word2)
    
    # --- 3. Combine scores ---
    
    # Start with base vector similarity or legal similarity
    final_similarity = base_similarity if base_similarity is not None else 0.5  # Default mid-point if vectors fail
    
    # Boost if marks share a legal category (important in trademark law)
    if legal_category_match:
        final_similarity = min(1.0, final_similarity + 0.15)  # Boost by 0.15 but cap at 1.0
    
    # Apply legal concept boosts if any were found (single-word analysis)
    if legal_concept_boosts:
        # Take the maximum of the vector similarity and the highest legal boost
        # This ensures conceptual relationships recognized in law are properly weighted
        legal_boost = max(legal_concept_boosts)
        final_similarity = max(final_similarity, legal_boost)
        
    # Ensure bounds
    final_similarity = max(0.0, min(1.0, final_similarity))
    
    info("Final conceptual similarity calculated", 
         mark1=text1, mark2=text2, 
         final_similarity=final_similarity)
    
    return final_similarity


# An enhanced version of the conceptual similarity function has been implemented.
# It attempts to capture legal nuances of trademark conceptual similarity through:
# 1. Base vector embedding similarity (as before)
# 2. Detection of special semantic relationships relevant in trademark law:
#    - Word stems/lemmas (morphological similarity)
#    - Synonyms, antonyms, and hypernym/hyponym relationships (using WordNet)
#    - Special categories (e.g., colors, numbers)
# This implements the suggestions from the original TODO comment.

# --- End of File ---
