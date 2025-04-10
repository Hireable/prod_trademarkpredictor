# Trademark Decision Intelligence AI Agent

A Google Cloud Function-based AI agent for analyzing trademark similarity and predicting opposition outcomes in UK/EU trademark law. Built using Google's Agent Development Kit (ADK) and deployed on Google Cloud Functions v2.

## Overview

This project implements an AI agent that:
1. Analyzes trademark similarity across multiple dimensions (visual, aural, conceptual, goods/services)
2. Predicts opposition outcomes based on similarity analysis
3. Provides detailed reasoning for its predictions

The system uses vector embeddings for semantic similarity comparisons and traditional algorithms (Levenshtein, phonetic matching) for wordmark comparisons.

## Project Structure

```
.
├── src/                    # Source code
│   ├── main.py            # Cloud Function entry point
│   ├── models.py          # Pydantic and SQLAlchemy models
│   └── tools/             # Agent tools and utilities
├── tests/                 # Test suite
├── docs/                  # Documentation
├── sft_jsonl/            # Training data
├── requirements.txt      # Python dependencies
├── .env.example         # Example environment variables
└── .cursorrules         # Cursor IDE configuration
```

## Key Components

### Data Models (`src/models.py`)

- **Pydantic Models**: For request/response validation and data transfer
  - `Trademark`: Core trademark representation
  - `GoodsService`: Goods/services items with NICE classification
  - `SimilarityScores`: Multi-dimensional similarity metrics
  - `PredictionResult`: Opposition outcome predictions

- **SQLAlchemy Models**: For database persistence
  - `TrademarkOrm`: Trademark database table
  - `GoodsServiceOrm`: Goods/services database table with vector embeddings

### Cloud Function (`src/main.py`)

- HTTP-triggered Cloud Function entry point
- Request validation using Pydantic models
- Error handling and response formatting
- Integration with Google ADK agent

## Prerequisites

1. Python 3.9+
2. Google Cloud Platform account with:
   - Cloud Functions v2 enabled
   - Appropriate IAM permissions
3. Supabase account and project

## Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd trademark-prediction-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration:
   # SUPABASE_URL=your_supabase_url
   # SUPABASE_KEY=your_supabase_key
   ```

5. Set up the database tables in Supabase:
   ```sql
   -- Enable the pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;

   -- Create the trademarks table
   CREATE TABLE trademarks (
       id SERIAL PRIMARY KEY,
       identifier VARCHAR NOT NULL UNIQUE,
       mark_text VARCHAR NOT NULL
   );

   -- Create the goods_services table
   CREATE TABLE goods_services (
       id SERIAL PRIMARY KEY,
       term VARCHAR NOT NULL,
       nice_class INTEGER NOT NULL,
       embedding FLOAT[],
       trademark_id INTEGER REFERENCES trademarks(id) ON DELETE CASCADE
   );

   -- Create indexes
   CREATE INDEX idx_trademarks_identifier ON trademarks(identifier);
   CREATE INDEX idx_goods_services_nice_class ON goods_services(nice_class);
   ```

## Development

### Local Development

1. Start the Cloud Function locally:
   ```bash
   functions-framework --target handle_request --debug
   ```

2. Run tests:
   ```bash
   pytest
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Include docstrings for all public modules, classes, and functions
- Use Pydantic models for data validation
- Follow async/await patterns for I/O operations

## Deployment

1. Build and deploy to Cloud Functions v2:
   ```bash
   gcloud functions deploy trademark-agent \
     --gen2 \
     --runtime=python39 \
     --region=europe-west2 \
     --source=. \
     --entry-point=handle_request \
     --trigger-http
   ```

## API Usage

### Similarity Analysis Request

```json
{
  "applicant_trademark": {
    "identifier": "UK000012345678",
    "wordmark": {
      "mark_text": "EXAMPLE"
    },
    "goods_services": [
      {
        "term": "Computer software",
        "nice_class": 9
      }
    ]
  },
  "opponent_trademark": {
    "identifier": "UK000087654321",
    "wordmark": {
      "mark_text": "EXEMPLAR"
    },
    "goods_services": [
      {
        "term": "Software as a service",
        "nice_class": 9
      }
    ]
  }
}
```

### Response Format

```json
{
  "message": "Successfully parsed request.",
  "applicant_identifier": "UK000012345678",
  "opponent_identifier": "UK000087654321"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[License details to be added]

## Acknowledgments

- Google Cloud Platform
- Google Agent Development Kit (ADK)
- Supabase
- Python-Levenshtein 