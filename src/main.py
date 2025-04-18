# src/main.py
"""
Google Cloud Function entry point for the Trademark AI Agent.

This function receives HTTP requests, parses the incoming JSON payload
into a Pydantic model, initializes the ADK agent, invokes the agent
to process the request, and returns the formatted response.
"""

import json
import os
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4 # Added for session management

import functions_framework
from flask import Request, Response
# Updated ADK imports
from google.adk.agents import Agent
from google.genai import Content, Part
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools import BaseTool # Aligning with documentation pattern
from pydantic import ValidationError
from fastapi import FastAPI, Depends
from google.adk.cli.fast_api import get_fast_api_app

# Import input/output models and the list of tools
from src.models import SimilarityTaskInput, SimilarityScores, Trademark, Wordmark, GoodsService
from src.tools.similarity_tools import trademark_similarity_tools # Import the list
from src.tools.prediction_tools import prediction_tools # Import prediction tools
from src.db import initialize_database
from src.logger import get_logger, info, warning, error, exception
from src.prompts.gemini_agent_prompt import get_adk_agent_prompt, format_analysis_prompt

# Initialize logger
logger = get_logger(__name__)

# Get the directory where main.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Example session DB URL (using SQLite for simplicity)
SESSION_DB_URL = "sqlite:///./sessions.db"

# Example allowed origins for CORS
ALLOWED_ORIGINS = ["*"]  # Adjust based on your security requirements

# Set web=False since we're using Cloud Functions
SERVE_WEB_INTERFACE = False

# Create toolkit with all available tools
trademark_tools = trademark_similarity_tools + prediction_tools

# Get the FastAPI app instance
app: FastAPI = get_fast_api_app(
    agent_dir=APP_DIR,
    session_db_url=SESSION_DB_URL,
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)

# Initialize the agent with tools
async def get_agent() -> Agent:
    """
    Creates and returns the ADK Agent with properly configured tools.
    Called as a FastAPI dependency.
    """
    # Create services for session management and artifact storage
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    
    # Get the optimized agent prompt with examples
    agent_prompt = get_adk_agent_prompt(include_examples=True)
    
    # Create the agent with tools and prompt template
    agent = Agent(
        tools=trademark_tools,
        session_service=session_service,
        artifact_service=artifact_service,
        default_prompt_context=agent_prompt,
    )
    
    return agent

# Set up startup event to run database initialization
@app.on_event("startup")
async def startup_db_client():
    """Initialize database on application startup."""
    try:
        logger.info("Running startup tasks")
        await initialize_database()
    except Exception as e:
        logger.exception("Error during startup", exc=e)
        # We log the error but don't raise to allow the service to start
        # even if DB initialization fails

# Set up shutdown event
@app.on_event("shutdown")
async def shutdown_db_client():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down application")
    # Add any cleanup logic here (closing sessions, etc.)

# --- Main Cloud Function Handler ---

@functions_framework.http
async def handle_request(request: Request) -> Response:
    """
    Cloud Function entry point that forwards requests to the FastAPI application.
    """
    # Convert Flask request to FastAPI request and handle it
    path = request.path
    if not path:
        path = "/"
        
    # Forward the request to the appropriate FastAPI endpoint
    if request.method == "POST" and path == "/":
        # Default to /run endpoint for POST requests to root
        path = "/run"
    
    # Get query parameters
    query_params = request.args.to_dict()
    
    # Get headers
    headers = {k: v for k, v in request.headers}
    
    # Get body
    body = request.get_data()
    
    # Log the incoming request
    logger.info(
        "Incoming request",
        method=request.method,
        path=path,
        content_length=len(body) if body else 0
    )
    
    # Create scope for FastAPI
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": request.method,
        "scheme": request.scheme,
        "path": path,
        "query_string": request.query_string,
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    
    # Handle the request through FastAPI
    async def receive():
        return {"type": "http.request", "body": body}
    
    async def send(message):
        if message["type"] == "http.response.start":
            nonlocal status_code, response_headers
            status_code = message["status"]
            response_headers = {k.decode(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            nonlocal response_body
            response_body = message["body"]
    
    status_code = 200
    response_headers = {}
    response_body = b""
    
    # Call FastAPI
    await app(scope, receive, send)
    
    # Log the response
    logger.info(
        "Sending response",
        status_code=status_code,
        content_length=len(response_body) if response_body else 0
    )
    
    # Convert to Flask response
    response = Response(
        response=response_body,
        status=status_code,
        headers=response_headers,
    )
    return response

# Define a custom ADK endpoint in FastAPI
@app.post("/analyze")
async def analyze_trademarks(
    data: SimilarityTaskInput,
    agent: Agent = Depends(get_agent)
):
    """
    Custom endpoint for trademark analysis.
    Provides a more structured API interface for direct integration.
    
    Args:
        data: The trademark comparison data.
        agent: ADK agent dependency.
    
    Returns:
        Analysis results with similarity scores and prediction.
    """
    try:
        # Log the request
        logger.info("Analyzing trademarks", 
                   applicant=data.applicant_trademark.identifier,
                   opponent=data.opponent_trademark.identifier)
        
        # Extract trademark details for the prompt
        applicant_mark = data.applicant_trademark.wordmark.mark_text
        opponent_mark = data.opponent_trademark.wordmark.mark_text
        
        # Extract classes and goods/services lists
        applicant_classes = [gs.nice_class for gs in data.applicant_trademark.goods_services]
        opponent_classes = [gs.nice_class for gs in data.opponent_trademark.goods_services]
        
        applicant_goods_services = [gs.term for gs in data.applicant_trademark.goods_services]
        opponent_goods_services = [gs.term for gs in data.opponent_trademark.goods_services]
        
        # Generate a structured analysis prompt
        analysis_prompt = format_analysis_prompt(
            applicant_mark=applicant_mark,
            opponent_mark=opponent_mark,
            applicant_classes=applicant_classes,
            opponent_classes=opponent_classes,
            applicant_goods_services=applicant_goods_services,
            opponent_goods_services=opponent_goods_services
        )
        
        # Prepare input for the agent
        session_id = str(uuid4())
        
        # Create a request content with the structured analysis prompt
        request_content = [
            Content(
                parts=[
                    Part(text=analysis_prompt),
                ],
                role="user",
            )
        ]
        
        # Create a runner and run the agent
        runner = Runner(agent)
        response = await runner.chat_async(
            session_id=session_id,
            message=request_content,
            tools=trademark_tools,
        )
        
        # Process and structure the agent's response
        analysis_result = {
            "message": "Trademark analysis completed",
            "applicant_identifier": data.applicant_trademark.identifier,
            "opponent_identifier": data.opponent_trademark.identifier,
            "analysis": response.text,
        }
        
        # Extract tool calls and results if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_results.append({
                    "tool": tool_call.name,
                    "input": tool_call.input,
                    "output": tool_call.output
                })
            analysis_result["tool_results"] = tool_results
        
        return analysis_result
    
    except ValidationError as e:
        logger.error("Validation error", error=str(e))
        return {"error": f"Invalid input data: {e}"}
    except Exception as e:
        logger.exception("Error processing trademark analysis", exc=e)
        return {"error": f"Error processing request: {str(e)}"}

# Define an endpoint for individual similarity calculations
@app.post("/calculate-similarity")
async def calculate_similarity(
    data: SimilarityTaskInput,
    similarity_type: str,
    agent: Agent = Depends(get_agent)
):
    """
    Endpoint for calculating individual similarity metrics between trademarks.
    
    Args:
        data: The trademark comparison data
        similarity_type: Type of similarity to calculate (visual, aural, conceptual, goods_services)
        agent: ADK agent dependency
        
    Returns:
        Similarity calculation result
    """
    valid_types = ["visual", "aural", "conceptual", "goods_services", "overall"]
    
    if similarity_type not in valid_types:
        return {"error": f"Invalid similarity type. Must be one of: {', '.join(valid_types)}"}
    
    try:
        # Create session ID
        session_id = str(uuid4())
        
        # Determine which tool to call based on similarity type
        if similarity_type == "visual":
            tool_name = "calculate_visual_wordmark_similarity_tool"
            tool_input = {
                "applicant_wordmark": data.applicant_trademark.wordmark,
                "opponent_wordmark": data.opponent_trademark.wordmark
            }
        elif similarity_type == "aural":
            tool_name = "calculate_aural_wordmark_similarity_tool"
            tool_input = {
                "applicant_wordmark": data.applicant_trademark.wordmark,
                "opponent_wordmark": data.opponent_trademark.wordmark
            }
        elif similarity_type == "conceptual":
            tool_name = "calculate_conceptual_wordmark_similarity_tool"
            tool_input = {
                "applicant_wordmark": data.applicant_trademark.wordmark,
                "opponent_wordmark": data.opponent_trademark.wordmark
            }
        elif similarity_type == "goods_services":
            tool_name = "calculate_goods_services_similarity_tool"
            tool_input = {
                "applicant_goods_services": data.applicant_trademark.goods_services,
                "opponent_goods_services": data.opponent_trademark.goods_services
            }
        elif similarity_type == "overall":
            # For overall similarity, we first need to calculate all individual similarities
            # and then pass them to the overall similarity calculator
            # This logic would need to be implemented
            return {"error": "Overall similarity calculation requires all other similarity scores first"}
        
        # Directly invoke the appropriate tool using runner.adk_tool_call_async
        runner = Runner(agent)
        response = await runner.adk_tool_call_async(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input
        )
        
        return {
            "similarity_type": similarity_type,
            "score": response,
            "applicant_identifier": data.applicant_trademark.identifier,
            "opponent_identifier": data.opponent_trademark.identifier
        }
    
    except Exception as e:
        logger.exception(f"Error calculating {similarity_type} similarity", exc=e)
        return {"error": f"Error calculating similarity: {str(e)}"}

# Define a prediction endpoint for opposition outcomes
@app.post("/predict")
async def predict_opposition_outcome(
    scores: SimilarityScores,
    data: SimilarityTaskInput,
    agent: Agent = Depends(get_agent)
):
    """
    Endpoint for predicting trademark opposition outcomes based on calculated similarity scores.
    
    Args:
        scores: The similarity scores calculated for the trademarks
        data: The trademark comparison data
        agent: ADK agent dependency
        
    Returns:
        Opposition prediction result
    """
    try:
        # Create session ID
        session_id = str(uuid4())
        
        # Create the prediction input
        prediction_input = {
            "prediction_task": {
                "applicant_trademark": data.applicant_trademark,
                "opponent_trademark": data.opponent_trademark,
                "similarity_scores": scores
            }
        }
        
        # Directly invoke the prediction tool
        runner = Runner(agent)
        prediction = await runner.adk_tool_call_async(
            session_id=session_id,
            tool_name="predict_opposition_outcome_tool",
            tool_input=prediction_input
        )
        
        return {
            "applicant_identifier": data.applicant_trademark.identifier,
            "opponent_identifier": data.opponent_trademark.identifier,
            "prediction": prediction
        }
    
    except Exception as e:
        logger.exception("Error predicting opposition outcome", exc=e)
        return {"error": f"Error predicting opposition outcome: {str(e)}"}

if __name__ == "__main__":
    # For local development
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)