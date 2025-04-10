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
from google.adk.types import Content, Part
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from pydantic import ValidationError
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

# Import input/output models and tools
from src.models import SimilarityTaskInput, SimilarityScores, Trademark, Wordmark, GoodsService
from src.tools.similarity_tools import similarity_toolset # Import the toolset instead of individual tools

# Get the directory where main.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Example session DB URL (using SQLite for simplicity)
SESSION_DB_URL = "sqlite:///./sessions.db"

# Example allowed origins for CORS
ALLOWED_ORIGINS = ["*"]  # Adjust based on your security requirements

# Set web=False since we're using Cloud Functions
SERVE_WEB_INTERFACE = False

# Get the FastAPI app instance
app: FastAPI = get_fast_api_app(
    agent_dir=APP_DIR,
    session_db_url=SESSION_DB_URL,
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)

# --- Helper Function to Get Tools ---

def get_agent_tools():
    """
    Retrieves the list of available ADK tools for the agent.

    Currently includes similarity calculation tools.

    Returns:
        A list of ADK Tool objects.
    """
    # Get all tools from the similarity toolset
    return similarity_toolset.get_tools()


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
    
    # Convert to Flask response
    response = Response(
        response=response_body,
        status=status_code,
        headers=response_headers,
    )
    return response

if __name__ == "__main__":
    # For local development
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)