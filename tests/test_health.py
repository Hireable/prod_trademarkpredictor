"""
Basic health check endpoint tests.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


def test_health_check_sync():
    """Test health check endpoint synchronously."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_check_async():
    """Test health check endpoint asynchronously using TestClient."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"} 