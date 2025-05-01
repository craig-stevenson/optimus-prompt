import os
import pytest

from optimus_prompt.core import Prompt
from optimus_prompt.providers import AnthropicProvider


@pytest.fixture
def api_key() -> str:
    """Get Anthropic API key from environment variable."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return key


@pytest.fixture
def provider(api_key: str) -> AnthropicProvider:
    """Create an instance of AnthropicProvider."""
    return AnthropicProvider(
        model="claude-3-5-sonnet-20240620",
        api_key=api_key
    )


@pytest.fixture
def prompt() -> Prompt:
    """Create a test prompt."""
    return Prompt(
        text="Tell me a random fact about space in exactly one sentence.",
        metadata={"test_type": "simple_generation"}
    )


def test_generate_response(
    provider: AnthropicProvider,
    prompt: Prompt
) -> None:
    """Test generating a response from Anthropic's API.
    
    This test:
    1. Sends a simple prompt to the API
    2. Verifies the response structure and content
    3. Checks that all expected metadata is present
    """
    # Generate response
    response = provider.generate(prompt)

    # Basic response validation
    assert response.text, "Response text should not be empty"
    assert isinstance(response.text, str), "Response text should be a string"
    assert len(response.text) > 0, "Response text should not be empty"

    # Provider info validation
    assert response.provider_name == "anthropic", "Provider name should be 'anthropic'"
    assert response.model == "claude-3-5-sonnet-20240620", "Model should be 'claude-3-5-sonnet-20240620'"

    # Metadata validation
    assert response.metadata is not None, "Response should have metadata"
    assert "latency" in response.metadata, "Response should include latency"
    assert isinstance(response.metadata["latency"], float), "Latency should be a float"
    assert response.metadata["latency"] > 0, "Latency should be positive"
    
    assert "message_id" in response.metadata, "Response should include message_id"
    assert isinstance(response.metadata["message_id"], str), "Message ID should be a string"
    
    assert "stop_reason" in response.metadata, "Response should include stop_reason"

    # Print response details for manual verification
    print("\nTest Response Details:")
    print(f"Text: {response.text}")
    print(f"Latency: {response.metadata['latency']:.2f}s")
    print(f"Message ID: {response.metadata['message_id']}")
    print(f"Stop Reason: {response.metadata['stop_reason']}")