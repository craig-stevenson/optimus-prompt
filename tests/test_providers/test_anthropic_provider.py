import os
import pytest
from unittest.mock import patch, MagicMock

from anthropic import APIError
from optimus_prompt.core import Prompt
from optimus_prompt.providers import AnthropicProvider, ProviderError


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


def test_list_available_models_success():
    """Test successfully listing available models."""
    # Mock the Anthropic client and models.list() response
    mock_model = MagicMock()
    mock_model.id = "claude-3-5-opus-20240620"
    mock_models = MagicMock()
    mock_models.data = [mock_model]
    
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value.models.list.return_value = mock_models
        
        # Call the method
        models = AnthropicProvider.list_available_models()
        
        # Verify results
        assert isinstance(models, list), "Should return a list"
        assert len(models) > 0, "Should return at least one model"
        assert "claude-3-5-opus-20240620" in models, "Should include expected model"
        assert all(m.startswith('claude-') for m in models), "All models should be Claude models"


def test_list_available_models_api_error():
    """Test fallback behavior when API call fails."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        # Simulate API error
        mock_anthropic.return_value.models.list.side_effect = APIError("API Error")
        
        # Call the method
        models = AnthropicProvider.list_available_models()
        
        # Verify fallback to CLAUDE_PRICING keys
        assert isinstance(models, list), "Should return a list"
        assert len(models) > 0, "Should return at least one model"
        assert "claude-3-5-opus-20240620" in models, "Should include models from CLAUDE_PRICING"
        assert "claude-3-5-sonnet-20240620" in models, "Should include models from CLAUDE_PRICING"


def test_list_available_models_unexpected_error():
    """Test error handling for unexpected exceptions."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        # Simulate unexpected error
        mock_anthropic.return_value.models.list.side_effect = Exception("Unexpected error")
        
        # Verify error handling
        with pytest.raises(ProviderError) as exc_info:
            AnthropicProvider.list_available_models()
        
        assert "Error fetching available models" in str(exc_info.value)