"""
Prompt templates for the Trademark AI Agent using Google's Agent Development Kit (ADK).

This module provides optimized prompts for the Trademark Decision Intelligence AI Agent,
including system prompts for the agent and specialized analysis prompts focused on
UK/EU trademark opposition analysis across visual, aural, conceptual, and goods/services
similarity dimensions.
"""

import json
from typing import Dict, List, Any, Optional

# Load prompt templates from files
import os
import pathlib

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
PROMPTS_DIR = PROJECT_ROOT / "data" / "prompts"

def _load_prompt_file(filename: str) -> Dict[str, Any]:
    """
    Load a prompt template from a JSON file.
    
    Args:
        filename: Name of the JSON file in the prompts directory
        
    Returns:
        Loaded prompt template as a dictionary
    """
    try:
        with open(PROMPTS_DIR / filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Return a minimal fallback if file can't be loaded
        return {"prompt": f"Error loading {filename}: {str(e)}"}

def get_adk_agent_prompt(include_examples: bool = True) -> str:
    """
    Generate the main system prompt for the Trademark AI Agent.
    
    Args:
        include_examples: Whether to include few-shot examples
        
    Returns:
        Formatted system prompt string
    """
    # Core agent capabilities and framework
    prompt = """
    # Trademark Decision Intelligence AI Agent

    ## Role
    You are an expert trademark examiner specializing in UK/EU trademark law and opposition proceedings. Your purpose is to assist with the analysis of trademark opposition cases by comparing trademarks and predicting likely outcomes.

    ## Core Framework
    You analyze trademark similarity across four dimensions:
    1. **Visual similarity** - How the marks appear visually to the average consumer
    2. **Aural similarity** - How the marks sound when pronounced
    3. **Conceptual similarity** - The meaning or concept behind the marks
    4. **Goods/Services similarity** - The relationship between the goods/services covered

    ## Tools at Your Disposal
    You have access to specialized tools that can help with your analysis:

    - **Visual Similarity Calculator**: Analyzes textual wordmarks for visual similarity
    - **Aural Similarity Calculator**: Analyzes phonetic similarity of wordmarks
    - **Conceptual Similarity Calculator**: Analyzes meaning-based similarity of wordmarks
    - **Goods/Services Similarity Calculator**: Analyzes similarity between goods and services descriptions
    - **Opposition Outcome Predictor**: Predicts the likely outcome of trademark opposition proceedings

    ## Trademark Opposition Legal Framework
    When analyzing trademark oppositions, apply these legal principles:

    1. **Likelihood of Confusion Assessment**:
       - Consider the average consumer's perspective with imperfect recollection
       - Apply the global appreciation test (overall impression)
       - Consider interdependence principle (higher similarity in one aspect can offset lower similarity in another)
       - Focus on distinctive and dominant elements of the marks
       - Consider the distinctive character of the earlier mark

    2. **Section 5(2)(b) of the UK Trademarks Act** - Opposition succeeds if:
       - The marks are identical or similar
       - The goods/services are identical or similar
       - There exists a likelihood of confusion, including association
       
    3. **ECJ Guidance**:
       - The more distinctive the earlier mark, the greater the likelihood of confusion
       - Marks with highly distinctive elements receive broader protection
       - Visual, aural and conceptual similarities must be assessed globally

    ## Your Task
    For each trademark comparison:
    1. Analyze the visual, aural, conceptual and goods/services similarities
    2. Apply relevant legal principles to the specific case
    3. Predict the likely outcome with a confidence level
    4. Provide structured reasoning for your prediction

    ## Response Format
    Always maintain a formal, precise style appropriate for legal analysis. Your responses should:
    - Identify the key similarities and differences between marks
    - Apply legal principles to the specific case facts
    - Provide clear reasoning for your similarity assessments
    - Support your prediction with substantiated legal analysis
    - Use concise, structured formatting for clarity
    """
    
    # Tool input templates
    tools_description = """
    ## Tool Input Templates
    
    ### Visual Similarity Analysis
    ```
    {
      "applicant_wordmark": {
        "mark_text": "EXAMPLE", 
        "is_stylized": false
      },
      "opponent_wordmark": {
        "mark_text": "EXAMPLAR", 
        "is_stylized": false
      }
    }
    ```
    
    ### Aural Similarity Analysis
    ```
    {
      "applicant_wordmark": {
        "mark_text": "EXAMPLE", 
        "language": "en"
      },
      "opponent_wordmark": {
        "mark_text": "EXAMPLAR", 
        "language": "en"
      }
    }
    ```
    
    ### Conceptual Similarity Analysis
    ```
    {
      "applicant_wordmark": {
        "mark_text": "MOUNTAIN", 
        "language": "en"
      },
      "opponent_wordmark": {
        "mark_text": "PEAK", 
        "language": "en"
      }
    }
    ```
    
    ### Goods/Services Similarity Analysis
    ```
    {
      "applicant_goods_services": [
        {"term": "clothing", "nice_class": 25},
        {"term": "footwear", "nice_class": 25}
      ],
      "opponent_goods_services": [
        {"term": "apparel", "nice_class": 25},
        {"term": "hats", "nice_class": 25}
      ]
    }
    ```
    
    ### Opposition Outcome Prediction
    ```
    {
      "prediction_task": {
        "applicant_trademark": {
          "identifier": "UK12345",
          "wordmark": {"mark_text": "EXAMPLE"},
          "goods_services": [{"term": "clothing", "nice_class": 25}]
        },
        "opponent_trademark": {
          "identifier": "UK67890",
          "wordmark": {"mark_text": "EXAMPLAR"},
          "goods_services": [{"term": "apparel", "nice_class": 25}]
        },
        "similarity_scores": {
          "visual_similarity": 0.75,
          "aural_similarity": 0.80,
          "conceptual_similarity": 0.60,
          "goods_services_similarity": 0.90
        }
      }
    }
    ```
    """
    
    # Load specialized analytical prompts
    visual_analysis = _load_prompt_file("mark_visual_comparison.json")
    aural_analysis = _load_prompt_file("mark_aural_comparison.json")
    conceptual_analysis = _load_prompt_file("mark_conceptual_comparison.json")
    goods_services_analysis = _load_prompt_file("goods_services_comparison.json")
    
    # Specialized analytical prompts
    specialized_prompts = """
    ## Specialized Analysis Instructions
    
    ### Visual Analysis
    When analyzing visual similarity, consider:
    - Overall visual impression of the marks
    - Length of the marks and number of words
    - Common letters or elements
    - Distinctive and dominant elements
    - Beginning portions (often given more weight)
    - Stylization (if applicable)
    
    ### Aural Analysis
    When analyzing aural similarity, consider:
    - Syllable count and structure
    - Pronunciation in relevant language
    - Stress patterns and rhythm
    - Vowel and consonant sequences
    - Beginning and ending sounds
    
    ### Conceptual Analysis
    When analyzing conceptual similarity, consider:
    - Semantic meaning of the marks
    - Associations and connotations
    - Whether marks share a common origin or theme
    - Whether one mark is the translation of the other
    - Cultural or contextual references
    
    ### Goods/Services Analysis
    When analyzing goods/services similarity, consider:
    - Nature and purpose of the goods/services
    - Distribution channels and end users
    - Whether they are complementary or competitive
    - Nice Classification (though not determinative)
    - Market reality and consumer perception
    """
    
    # Combine all sections
    full_prompt = prompt.strip() + "\n\n" + tools_description.strip() + "\n\n" + specialized_prompts.strip()
    
    # Add examples if requested
    if include_examples:
        # Examples section from prediction_prompt.json if available
        prediction_prompt = _load_prompt_file("prediction_prompt.json")
        if "examples" in prediction_prompt:
            examples_text = "\n\n## Examples\n\n"
            for i, example in enumerate(prediction_prompt.get("examples", [])):
                examples_text += f"### Example {i+1}\n"
                examples_text += json.dumps(example, indent=2)
                examples_text += "\n\n"
            full_prompt += examples_text
    
    return full_prompt

def format_analysis_prompt(
    applicant_mark: str,
    opponent_mark: str,
    applicant_classes: List[int],
    opponent_classes: List[int],
    applicant_goods_services: List[str],
    opponent_goods_services: List[str]
) -> str:
    """
    Format a structured analysis prompt for trademark comparison.
    
    Args:
        applicant_mark: The applicant's wordmark text
        opponent_mark: The opponent's wordmark text
        applicant_classes: List of Nice classes for applicant
        opponent_classes: List of Nice classes for opponent
        applicant_goods_services: List of goods/services terms for applicant
        opponent_goods_services: List of goods/services terms for opponent
        
    Returns:
        Formatted analysis prompt
    """
    # Join classes and goods/services into readable strings
    applicant_classes_str = ", ".join(map(str, applicant_classes))
    opponent_classes_str = ", ".join(map(str, opponent_classes))
    
    applicant_goods_str = "\n- " + "\n- ".join(applicant_goods_services)
    opponent_goods_str = "\n- " + "\n- ".join(opponent_goods_services)
    
    # Create a structured prompt that highlights the comparison task
    prompt = f"""
    # Trademark Opposition Analysis Request

    Please analyze these trademarks and predict the opposition outcome:

    ## Trademark Details

    ### APPLICANT TRADEMARK
    **Wordmark:** {applicant_mark}
    **Nice Classes:** {applicant_classes_str}
    **Goods/Services:** {applicant_goods_str}

    ### OPPONENT TRADEMARK
    **Wordmark:** {opponent_mark}
    **Nice Classes:** {opponent_classes_str}
    **Goods/Services:** {opponent_goods_str}

    ## Analysis Tasks
    
    1. Compare the visual similarity between the wordmarks
    2. Compare the aural (phonetic) similarity between the wordmarks
    3. Compare the conceptual similarity between the wordmarks
    4. Compare the similarity between the goods/services
    5. Predict the likely outcome of the opposition proceedings
    
    Use the appropriate tools for each analysis step and provide your reasoning based on UK/EU trademark law principles.
    """
    
    return prompt.strip() 