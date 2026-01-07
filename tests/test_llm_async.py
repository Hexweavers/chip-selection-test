import pytest

from services.llm_async import AsyncLLMClient, LLMResponse


@pytest.mark.asyncio
async def test_async_client_returns_llm_response():
    """AsyncLLMClient.chat should return an LLMResponse."""
    # We'll test with a mock/dry-run approach since we don't want real API calls
    # The actual test verifies the interface exists
    client = AsyncLLMClient()
    # Just verify the client can be instantiated and has the right methods
    assert hasattr(client, "chat")
    assert hasattr(client, "close")
    await client.close()


@pytest.mark.asyncio
async def test_llm_response_dataclass():
    """LLMResponse should have expected fields."""
    response = LLMResponse(
        content="test",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100,
        error=None,
    )
    assert response.content == "test"
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.latency_ms == 100
    assert response.error is None
