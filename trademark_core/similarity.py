"""
Core similarity calculation functions for trademark comparison.

This module provides functions to calculate visual, aural, conceptual, and goods/services
similarity between trademarks, operating entirely in memory without database dependencies.
"""

from typing import List, Optional, Dict, Tuple
import Levenshtein
from metaphone import doublemetaphone
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

from trademark_core import models

# Initialize NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize models
_sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
_legal_bert_tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
_legal_bert_model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')

def calculate_visual_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate visual similarity between two trademarks using Levenshtein distance.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    # Clean and normalize marks
    mark1 = mark1.lower().strip()
    mark2 = mark2.lower().strip()
    
    # Handle empty strings
    if not mark1 and not mark2:
        return 1.0
    if not mark1 or not mark2:
        return 0.0
    
    # Calculate Levenshtein ratio
    return Levenshtein.ratio(mark1, mark2)

def calculate_aural_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate aural (phonetic) similarity between two trademarks using Double Metaphone.
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    # Clean and normalize marks
    mark1 = mark1.lower().strip()
    mark2 = mark2.lower().strip()
    
    # Handle empty strings
    if not mark1 and not mark2:
        return 1.0
    if not mark1 or not mark2:
        return 0.0
    
    # Get Double Metaphone codes
    code1_primary, code1_alt = doublemetaphone(mark1)
    code2_primary, code2_alt = doublemetaphone(mark2)
    
    # Calculate similarities using primary and alternate codes
    primary_sim = Levenshtein.ratio(code1_primary, code2_primary)
    
    # If alternates exist, consider them too
    alt_sims = []
    if code1_alt and code2_alt:
        alt_sims.append(Levenshtein.ratio(code1_alt, code2_alt))
    if code1_primary and code2_alt:
        alt_sims.append(Levenshtein.ratio(code1_primary, code2_alt))
    if code1_alt and code2_primary:
        alt_sims.append(Levenshtein.ratio(code1_alt, code2_primary))
    
    # Return highest similarity found
    return max([primary_sim] + alt_sims) if alt_sims else primary_sim

async def calculate_conceptual_similarity(mark1: str, mark2: str) -> float:
    """
    Calculate conceptual similarity between two trademarks using a combination of:
    1. Semantic embeddings from SentenceTransformer
    2. Legal domain understanding from LegalBERT
    3. WordNet-based conceptual relationships
    
    Args:
        mark1: First trademark text
        mark2: Second trademark text
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    # Clean and normalize marks
    mark1 = mark1.lower().strip()
    mark2 = mark2.lower().strip()
    
    # Handle empty strings
    if not mark1 and not mark2:
        return 1.0
    if not mark1 or not mark2:
        return 0.0
    
    # 1. Get general semantic similarity using SentenceTransformer
    embeddings1 = _sentence_transformer.encode([mark1], convert_to_tensor=True)
    embeddings2 = _sentence_transformer.encode([mark2], convert_to_tensor=True)
    semantic_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)[0].item()
    
    # 2. Get legal domain similarity using LegalBERT
    inputs1 = _legal_bert_tokenizer(mark1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = _legal_bert_tokenizer(mark2, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs1 = _legal_bert_model(**inputs1)
        outputs2 = _legal_bert_model(**inputs2)
    
    legal_embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    legal_embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    legal_sim = torch.nn.functional.cosine_similarity(legal_embeddings1, legal_embeddings2)[0].item()
    
    # 3. Get WordNet-based conceptual similarity
    def get_synsets(word: str) -> List:
        return wordnet.synsets(word)
    
    def get_conceptual_similarity(word1: str, word2: str) -> float:
        synsets1 = get_synsets(word1)
        synsets2 = get_synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        # Get max similarity between any pair of synsets
        max_sim = 0.0
        for syn1 in synsets1:
            for syn2 in synsets2:
                sim = syn1.path_similarity(syn2) or 0.0
                max_sim = max(max_sim, sim)
        
        return max_sim
    
    # Split compound marks and get max conceptual similarity
    words1 = mark1.split()
    words2 = mark2.split()
    
    conceptual_sims = []
    for w1 in words1:
        for w2 in words2:
            sim = get_conceptual_similarity(w1, w2)
            conceptual_sims.append(sim)
    
    wordnet_sim = max(conceptual_sims) if conceptual_sims else 0.0
    
    # Combine similarities with weights
    weights = {
        'semantic': 0.4,
        'legal': 0.4,
        'wordnet': 0.2
    }
    
    combined_sim = (
        weights['semantic'] * semantic_sim +
        weights['legal'] * legal_sim +
        weights['wordnet'] * wordnet_sim
    )
    
    return max(0.0, min(1.0, combined_sim))

def calculate_goods_services_similarity(applicant_gs: List[models.GoodService], 
                                     opponent_gs: List[models.GoodService]) -> float:
    """
    Calculate similarity between two lists of goods/services using semantic similarity
    and Nice class matching.
    
    Args:
        applicant_gs: List of applicant's goods/services
        opponent_gs: List of opponent's goods/services
        
    Returns:
        float: Similarity score between 0.0 (dissimilar) and 1.0 (identical)
    """
    if not applicant_gs or not opponent_gs:
        return 0.0
    
    # Get embeddings for all terms
    applicant_embeddings = _sentence_transformer.encode(
        [gs.term for gs in applicant_gs],
        convert_to_tensor=True
    )
    opponent_embeddings = _sentence_transformer.encode(
        [gs.term for gs in opponent_gs],
        convert_to_tensor=True
    )
    
    # Calculate similarity matrix
    similarity_matrix = torch.nn.functional.cosine_similarity(
        applicant_embeddings.unsqueeze(1),
        opponent_embeddings.unsqueeze(0),
        dim=2
    )
    
    # For each applicant term, find the most similar opponent term
    max_similarities = []
    for i, app_gs in enumerate(applicant_gs):
        best_sim = 0.0
        for j, opp_gs in enumerate(opponent_gs):
            # Base similarity from embeddings
            term_sim = similarity_matrix[i][j].item()
            
            # Adjust based on Nice class matching
            class_match = 1.0 if app_gs.nice_class == opp_gs.nice_class else 0.5
            
            # Combine term and class similarity
            combined_sim = term_sim * class_match
            best_sim = max(best_sim, combined_sim)
        
        max_similarities.append(best_sim)
    
    # Return average of best similarities
    return sum(max_similarities) / len(max_similarities)

async def calculate_overall_similarity(mark1: models.Mark, mark2: models.Mark) -> models.MarkComparison:
    """
    Calculate overall similarity between two trademarks across all dimensions.
    
    Args:
        mark1: First trademark
        mark2: Second trademark
        
    Returns:
        MarkComparison: Comparison results for all dimensions
    """
    # Calculate individual similarities
    visual_sim = calculate_visual_similarity(mark1.wordmark, mark2.wordmark)
    aural_sim = calculate_aural_similarity(mark1.wordmark, mark2.wordmark)
    conceptual_sim = await calculate_conceptual_similarity(mark1.wordmark, mark2.wordmark)
    
    # Map float scores to EnumStr values
    def score_to_enum(score: float) -> models.EnumStr:
        if score > 0.9:
            return "identical"
        elif score > 0.7:
            return "high"
        elif score > 0.5:
            return "moderate"
        elif score > 0.3:
            return "low"
        else:
            return "dissimilar"
    
    # Convert scores to enums
    visual = score_to_enum(visual_sim)
    aural = score_to_enum(aural_sim)
    conceptual = score_to_enum(conceptual_sim)
    
    # Calculate overall similarity with weights
    weights = {
        'visual': 0.40,
        'aural': 0.35,
        'conceptual': 0.25
    }
    
    overall_score = (
        weights['visual'] * visual_sim +
        weights['aural'] * aural_sim +
        weights['conceptual'] * conceptual_sim
    )
    
    overall = score_to_enum(overall_score)
    
    return models.MarkComparison(
        visual=visual,
        aural=aural,
        conceptual=conceptual,
        overall=overall
    ) 