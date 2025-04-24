"""
Tests for the Trademark AI LLM integration module.

This module tests the LLM integration with Vertex AI and the Gemini 2.5 Pro model
for generating detailed reasoning based on trademark similarity analysis.
"""

import pytest
from unittest.mock import patch, MagicMock

from google import genai
from google.api_core.exceptions import GoogleAPIError

from trademark_core import models
from trademark_core.llm import generate_prediction_reasoning


class TestLLMIntegration:
    """Tests for LLM integration with Vertex AI."""
    
    @pytest.fixture
    def mock_request(self):
        """Create a sample opposition request object for testing."""
        return MagicMock(
            applicant=models.Mark(wordmark="TESTMARK", is_registered=False),
            opponent=models.Mark(wordmark="TESTMARKS", is_registered=True, registration_number="12345"),
            applicant_goods=[models.GoodService(term="Computer software", nice_class=9)],
            opponent_goods=[models.GoodService(term="Software as a service", nice_class=42)]
        )
    
    @pytest.fixture
    def mark_comparison(self):
        """Create a sample mark comparison object for testing."""
        return models.MarkComparison(
            visual="high",
            aural="high", 
            conceptual="moderate",
            overall="high"
        )
        
    @pytest.mark.asyncio
    async def test_generate_prediction_reasoning_success(self, mock_request, mark_comparison):
        """Test successfully generating prediction reasoning."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "This is a test reasoning response from the LLM."
        mock_response.candidates = []
        
        # Setup mock client
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        # Patch the genai.Client to return our mock
        with patch('trademark_core.llm.client', mock_client):
            # Call the function
            result = await generate_prediction_reasoning(
                mark_comparison=mark_comparison,
                goods_similarity=0.8,
                likelihood_of_confusion=True,
                request=mock_request
            )
            
            # Assertions
            assert isinstance(result, str)
            assert "test reasoning response" in result
            mock_client.models.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_prediction_reasoning_with_function_calls(
        self, mock_request, mark_comparison
    ):
        """Test generating prediction reasoning with function calls."""
        # Setup mock for initial response with function calls
        mock_initial_response = MagicMock()
        mock_initial_response.text = "Initial response with function calls"
        mock_function_call = MagicMock()
        mock_function_call.name = "calculate_visual_similarity"
        mock_function_call.args = {"mark1": "TESTMARK", "mark2": "TESTMARKS"}
        
        mock_content_part = MagicMock()
        mock_content_part.function_call = mock_function_call
        
        mock_candidate_content = MagicMock()
        mock_candidate_content.parts = [mock_content_part]
        
        mock_candidate = MagicMock()
        mock_candidate.content = mock_candidate_content
        
        mock_initial_response.candidates = [mock_candidate]
        
        # Create mock final response
        mock_final_response = MagicMock()
        mock_final_response.text = "Final response with function call results integrated."
        mock_final_response.candidates = []
        
        # Setup mock client
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_final_response  # We only need one response now
        
        # Patch the genai.Client to return our mock
        with patch('trademark_core.llm.client', mock_client):
            # Call the function
            result = await generate_prediction_reasoning(
                mark_comparison=mark_comparison,
                goods_similarity=0.8,
                likelihood_of_confusion=True,
                request=mock_request
            )
            
            # Assertions
            assert isinstance(result, str)
            assert "Final response with function call results" in result
            assert mock_client.models.generate_content.call_count == 1
    
    @pytest.mark.asyncio
    async def test_generate_prediction_reasoning_api_error(
        self, mock_request, mark_comparison
    ):
        """Test that API errors are properly raised."""
        # Setup mock client to raise API error
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = GoogleAPIError("API Error")
        
        # Patch the genai.Client to return our mock
        with patch('trademark_core.llm.client', mock_client):
            # Call the function and expect it to raise GoogleAPIError
            with pytest.raises(GoogleAPIError) as exc_info:
                await generate_prediction_reasoning(
                    mark_comparison=mark_comparison,
                    goods_similarity=0.8,
                    likelihood_of_confusion=True,
                    request=mock_request
                )
            
            assert str(exc_info.value) == "API Error" 