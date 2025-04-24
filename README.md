# Trademark Similarity Prediction API

A FastAPI service that predicts the likelihood of success in trademark opposition cases by analyzing:
- Visual similarity between marks
- Aural (phonetic) similarity
- Conceptual similarity
- Goods/services similarity
- Overall likelihood of confusion

## Features

- **Mark Similarity Analysis**
  - Visual similarity using Levenshtein distance
  - Aural similarity using Double Metaphone algorithm
  - Conceptual similarity using a combination of:
    - Sentence transformers for semantic understanding
    - LegalBERT for legal domain knowledge
    - WordNet for conceptual relationships

- **Goods/Services Analysis**
  - Semantic similarity between terms
  - Nice class matching
  - Trade channel consideration

- **LLM-Powered Reasoning**
  - Detailed legal analysis using Google's Gemini 2.5 Pro via Vertex AI
  - Function calling allows the LLM to dynamically invoke similarity algorithms
  - Professional explanation of key factors with UK/EU legal principles
  - Clear prediction of opposition outcome

## Installation

1. Install Python 3.10 or later
2. Clone this repository
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables:
   ```bash
   cp env.example .env
   ```
   Edit `.env` to add your Google Cloud Project ID and Gemini API Key

## Usage

1. Start the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Use the `/predict` endpoint to analyze trademark similarity:
   ```json
   POST /predict
   {
     "applicant": {
       "wordmark": "ACME",
       "is_registered": false
     },
     "opponent": {
       "wordmark": "ACMEE",
       "is_registered": true,
       "registration_number": "12345"
     },
     "applicant_goods": [
       {
         "term": "Computer software",
         "nice_class": 9
       }
     ],
     "opponent_goods": [
       {
         "term": "Software as a service",
         "nice_class": 42
       }
     ]
   }
   ```

4. The response will include:
   - Mark comparison results (visual, aural, conceptual)
   - Goods/services similarity score
   - Likelihood of confusion prediction
   - Detailed legal reasoning

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

The API requires the following environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| GOOGLE_CLOUD_PROJECT | Google Cloud project ID | Yes |
| GOOGLE_CLOUD_LOCATION | Google Cloud region (e.g., us-central1) | Yes |
| GOOGLE_API_KEY | Gemini API key | Yes |
| GOOGLE_APPLICATION_CREDENTIALS | Path to service account key (alternative auth method) | No |
| PORT | Port for FastAPI server (default: 8000) | No |
| HOST | Host for FastAPI server (default: 0.0.0.0) | No |

## Development

- Run tests: `pytest`
- Run linter: `ruff check .`
- Format code: `ruff format .`

## License

MIT