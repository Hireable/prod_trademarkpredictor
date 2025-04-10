# main.py
"""
Entry point for Google Cloud Functions v2.
Imports and exports the handle_request function from src/main.py.
"""

# Import the main handler from the src module
from src.main import handle_request

# This exports the handle_request function for Cloud Functions
# No need to redefine the function as we're directly exporting it

# If needed, add any additional global initialization here 