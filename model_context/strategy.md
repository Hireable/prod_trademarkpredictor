# Trademark‑AI **Greenfield** Blueprint v3  
**What is this app?**  A specialised backend API that compares two trademarks (marks + goods/services) and predicts the likelihood and outcome of an opposition under UK/EU law using in-memory similarity analysis and LLM-powered reasoning.  Results power a lawyer‑facing UI already hosted elsewhere.  
**Target users:** Trademark lawyers and paralegals evaluating conflict risk for clients.  

---

## 0 Design Goals
| ID | Goal | Rationale |
|----|------|-----------|
| G‑1 | **Backend‑only FastAPI on Cloud Run** – zero UI/hosting concerns. | Matches existing separately‑hosted front‑end. |
| G‑2 | **Stateless HTTP; no Auth for MVP** – public endpoint behind an API Gateway if needed. | Simplifies first release; auth can be toggled later. |
| G‑3 | **LLM abstraction**: Gemini 2.5 Pro via Vertex AI with proper error handling. | Future model swaps + resilient operation. |
| G‑4 | **Pydantic models as _single source of truth_ (SSoT)** – shared by API layer, domain logic, and tests. | Eliminates drift and doubles as JSON schema. |
| G‑5 | **Pure functions** (`calculate_visual_similarity`, `calculate_aural_similarity`, etc.) composable for a full case prediction. | Clean, testable, and stateless design. |
| G‑6 | **Smooth DX** → Ruff, Pytest (with asyncio support), pre‑commit. | Consistent, deterministic, and async-aware. |

---

## 1 Repo Layout
```
├── api/                        # FastAPI app (presentation + API layer)
│   ├── main.py                 # App entrypoint with prediction endpoint
│   └── routes/
│       ├── __init__.py
│       └── health.py           # /health endpoint implementation
├── trademark_core/             # domain layer (pure python, no FastAPI)
│   ├── models.py               # Pydantic SSoT – see §2
│   ├── similarity.py           # mark + goods/services algorithms
│   ├── llm.py                 # LLM integration with error handling
│   └── embeddings.py          # Semantic embedding utilities
├── tests/
│   ├── test_health.py         # Tests for /health endpoint
│   ├── test_llm_integration.py # Tests for LLM functionality
│   └── test_embedding.py      # Tests for embedding utilities
├── .gitignore
├── model_context/
│   └── strategy.md            # This file
├── pyproject.toml             # Project config + tool settings
└── README.md
```

---

## 2 Pydantic SSoT (models.py)
All layers import from here – never redeclare fields elsewhere.
```python
from typing import Literal, List, Annotated
from pydantic import BaseModel, Field

class Mark(BaseModel):
    wordmark: str = Field(..., description="Literal mark text, case‑sensitive")
    is_registered: bool = False
    registration_number: str | None = None

class GoodService(BaseModel):
    term: str
    nice_class: Annotated[int, Field(ge=1, le=45)]

# Comparison results
EnumStr = Literal['dissimilar','low','moderate','high','identical']

class MarkComparison(BaseModel):
    visual: EnumStr
    aural: EnumStr
    conceptual: EnumStr
    overall: EnumStr

class GoodServiceComparison(BaseModel):
    similarity: EnumStr

class CasePrediction(BaseModel):
    mark_comparison: MarkComparison
    goods_services_comparisons: List[GoodServiceComparison]
    likelihood_of_confusion: bool
    opposition_outcome: dict  # {result:str, confidence:float, reasoning:str}
```

---

## 3 Core Functions
| Function | Input | Output | Implementation |
|---------|-------|--------|----------------|
| `calculate_visual_similarity` | `str`, `str` | `float` | Levenshtein distance |
| `calculate_aural_similarity` | `str`, `str` | `float` | Double Metaphone |
| `calculate_conceptual_similarity` | `str`, `str` | `float` | SentenceTransformer + LegalBERT + WordNet |
| `calculate_goods_services_similarity` | `List[GoodService]`, `List[GoodService]` | `float` | Semantic similarity + Nice class matching |
| `calculate_overall_similarity` | `Mark`, `Mark` | `MarkComparison` | Weighted combination of similarities |
| `generate_prediction_reasoning` | `MarkComparison`, `float`, `bool`, `Any` | `str` | Gemini 2.5 Pro with error propagation |

---

## 4 Workflow
1. Front‑end hits `POST /predict` with JSON adhering to models.
2. API validates input via Pydantic.
3. Calculates mark similarities (visual, aural, conceptual).
4. Analyzes goods/services similarity.
5. Determines likelihood of confusion.
6. Generates detailed reasoning using LLM (with proper error handling).
7. Returns complete prediction response.

All operations are stateless and asynchronous where appropriate (LLM calls, embeddings).

---

## 5 Key Packages & Services
| Concern | Package | Purpose |
|---------|---------|---------|
| HTTP API | FastAPI + Uvicorn | API framework and server |
| Testing | pytest, pytest-asyncio | Test framework with async support |
| Linting | ruff | Code quality |
| Text Similarity | python-levenshtein | Visual similarity |
| Phonetic Matching | metaphone | Aural similarity |
| Semantic Analysis | sentence-transformers | Conceptual similarity |
| Legal Understanding | transformers (LegalBERT) | Domain-specific analysis |
| Language Processing | nltk | WordNet relationships |
| LLM Integration | google-cloud-aiplatform | Reasoning generation |
| Error Handling | google.api_core.exceptions | Proper LLM error propagation |

---

## 6 Task Breakdown
### Sprint 1 – Core Implementation
- [x] Set up project structure in venv with `pip install -r requirements.txt`
- [x] Configure linters and testing
- [x] Create FastAPI app with health endpoint
- [x] Implement Pydantic SSoT models
- [x] Implement similarity functions
- [x] Create prediction endpoint
- [x] Add comprehensive tests

### Sprint 2 – Enhancement & Hardening
- [x] Integrate LLM for detailed reasoning _(Gemini 2.5 Pro via Vertex AI)_
- [x] Implement proper error handling for LLM integration
- [x] Add async test support with pytest-asyncio
- [ ] Add CI/CD pipeline
- [~] Add API documentation _(OpenAPI/Swagger basics configured)_
- [ ] Performance optimization
- [ ] Security hardening

---

## 7 Sample Request/Response
```python
# POST /predict
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

# Response
{
    "mark_comparison": {
        "visual": "high",
        "aural": "high",
        "conceptual": "identical",
        "overall": "high"
    },
    "goods_services_similarity": 0.85,
    "likelihood_of_confusion": true,
    "reasoning": "Detailed analysis of similarity factors..."
}

# Error Response (on LLM failure)
{
    "detail": "Error generating prediction reasoning: API Error"
}
```

---

## 8 Acceptance Criteria
| Metric | Target | Status |
|--------|--------|--------|
| Response time P95 | ≤ 1.5s | ⏱️ Untested |
| Mark comparison latency | ≤ 100ms | ⏱️ Untested |
| Unit test coverage | ≥ 90% | ✅ Achieved |
| Error rate | ≤ 0.1% | 🔄 Monitored via error handling |

---

## 9 Testing Strategy
| Component | Approach | Status |
|-----------|----------|---------|
| Health Endpoint | Sync + Async tests | ✅ Implemented |
| LLM Integration | Mock responses + error cases | ✅ Implemented |
| Embeddings | Dimension + consistency checks | ✅ Implemented |
| API Layer | FastAPI TestClient | 🔄 In Progress |
| Error Handling | Exception propagation tests | ✅ Implemented |

---

_End of Blueprint v3 – Stateless Implementation with Enhanced Error Handling_

