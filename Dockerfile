# Use the official Python 3.11 slim image as a base
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc (equivalent to python -B)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1
# Set the port the container will listen on. Cloud Run injects its own PORT env var (default 8080)
ENV PORT 8080

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file
COPY requirements.txt .

# Install dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a non-root user and switch to it for security
# RUN adduser --disabled-password --gecos "" appuser
# USER appuser
# Note: Cloud Run's sandbox runs as non-root anyway, so creating
# a specific user isn't strictly necessary unless required by other tools.
# Keeping it simple for now.

# Expose the port the app runs on
EXPOSE $PORT

# Define the command to run the application
# Assumes your FastAPI app instance is named 'app' in 'src/main.py'
# Uses the PORT environment variable defined above (or injected by Cloud Run)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "$PORT"] 