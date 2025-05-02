import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from openai import OpenAIError
from optimus_prompt.core import Prompt
from optimus_prompt.providers import OpenAIProvider
from optimus_prompt.providers.openai_provider import GPT_PRICING, ProviderError


@pytest.fixture
def api_key() -> str:
    """Get OpenAI API key from environment variable."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return key


@pytest.fixture
def provider(api_key: str) -> OpenAIProvider:
    """Create an instance of OpenAIProvider."""
    return OpenAIProvider(
        model="gpt-3.5-turbo",
        api_key=api_key
    )


@pytest.fixture
def prompt() -> Prompt:
    """Create a test prompt."""
    return Prompt(
        text="Tell me a random fact about space in exactly one sentence.",
        metadata={"test_type": "simple_generation"}
    )


def test_provider_initialization() -> None:
    """Test provider initialization with different configurations."""
    # Test with explicit API key
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="test-key")
    assert provider.model == "gpt-3.5-turbo"
    assert provider.name == "openai"

    # Test without API key (should use environment variable)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        provider = OpenAIProvider(model="gpt-3.5-turbo")
        assert provider.model == "gpt-3.5-turbo"


def test_generate_response(
    provider: OpenAIProvider,
    prompt: Prompt
) -> None:
    """Test generating a response from OpenAI's API.
    
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
    assert response.provider_name == "openai", "Provider name should be 'openai'"
    assert response.model == "gpt-3.5-turbo", "Model should match initialized model"

    # Metadata validation
    assert response.metadata is not None, "Response should have metadata"
    assert "latency" in response.metadata, "Response should include latency"
    assert isinstance(response.metadata["latency"], float), "Latency should be a float"
    assert response.metadata["latency"] > 0, "Latency should be positive"
    
    assert "token_count" in response.metadata, "Response should include token count"
    assert "prompt_tokens" in response.metadata, "Response should include prompt tokens"
    assert "completion_tokens" in response.metadata, "Response should include completion tokens"
    assert "finish_reason" in response.metadata, "Response should include finish reason"
    assert "cost" in response.metadata, "Response should include cost information"

    # Print response details for manual verification
    print("\nTest Response Details:")
    print(f"Text: {response.text}")
    print(f"Latency: {response.metadata['latency']:.2f}s")
    print(f"Tokens: {response.metadata['token_count']}")
    print(f"Cost: ${response.metadata['cost']['total']:.6f}")


def test_calculate_cost() -> None:
    """Test cost calculation for different models and token counts."""
    provider = OpenAIProvider(model="gpt-4")
    
    # Test GPT-4 cost calculation
    tokens = {"input_tokens": 100, "output_tokens": 50}
    cost = provider.calculate_cost(tokens)
    expected_cost = (
        (100 / 1_000_000) * GPT_PRICING["gpt-4"]["input_per_million"] +
        (50 / 1_000_000) * GPT_PRICING["gpt-4"]["output_per_million"]
    )
    assert cost == round(expected_cost, 6)

    # Test GPT-3.5-turbo cost calculation
    provider.model = "gpt-3.5-turbo"
    cost = provider.calculate_cost(tokens)
    expected_cost = (
        (100 / 1_000_000) * GPT_PRICING["gpt-3.5-turbo"]["input_per_million"] +
        (50 / 1_000_000) * GPT_PRICING["gpt-3.5-turbo"]["output_per_million"]
    )
    assert cost == round(expected_cost, 6)

    # Test unknown model falls back to gpt-3.5-turbo pricing
    provider.model = "unknown-model"
    cost = provider.calculate_cost(tokens)
    expected_cost = (
        (100 / 1_000_000) * GPT_PRICING["gpt-3.5-turbo"]["input_per_million"] +
        (50 / 1_000_000) * GPT_PRICING["gpt-3.5-turbo"]["output_per_million"]
    )
    assert cost == round(expected_cost, 6)


def test_get_model_info() -> None:
    """Test model information retrieval and caching."""
    provider = OpenAIProvider(model="gpt-4")
    
    # Get model info
    info = provider.get_model_info()
    
    # Verify structure and content
    assert info["name"] == "gpt-4"
    assert info["provider"] == "openai"
    assert isinstance(info["capabilities"], list)
    assert "text" in info["capabilities"]
    assert "code" in info["capabilities"]
    assert "analysis" in info["capabilities"]
    assert info["max_tokens"] == 8192

    # Test caching - should return same object
    info2 = provider.get_model_info()
    assert info is info2, "Model info should be cached"

    # Test different model
    provider = OpenAIProvider(model="gpt-3.5-turbo")
    info = provider.get_model_info()
    assert info["name"] == "gpt-3.5-turbo"
    assert info["max_tokens"] == 4096


@patch('openai.OpenAI')
def test_error_handling(mock_openai: Mock) -> None:
    """Test error handling in the provider."""
    provider = OpenAIProvider(model="gpt-3.5-turbo", api_key="test-key")
    prompt = Prompt(text="Test prompt")

    # Test API error
    mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
    
    with pytest.raises(ProviderError) as exc_info:
        provider.generate(prompt)
    assert "Error generating response from OpenAI" in str(exc_info.value)


def test_list_available_models_success():
    """Test successfully listing available models."""
    # Mock the OpenAI client and models.list() response
    mock_model1 = MagicMock()
    mock_model1.id = "gpt-4"
    mock_model2 = MagicMock()
    mock_model2.id = "gpt-3.5-turbo"
    mock_model3 = MagicMock()
    mock_model3.id = "text-davinci-003"  # Should be filtered out
    
    mock_models = MagicMock()
    mock_models.data = [mock_model1, mock_model2, mock_model3]
    
    with patch('openai.OpenAI') as mock_openai:
        mock_openai.return_value.models.list.return_value = mock_models
        
        # Call the method
        models = OpenAIProvider.list_available_models()
        
        # Verify results
        assert isinstance(models, list), "Should return a list"
        assert len(models) == 2, "Should return only GPT models"
        assert "gpt-4" in models, "Should include GPT-4"
        assert "gpt-3.5-turbo" in models, "Should include GPT-3.5"
        assert "text-davinci-003" not in models, "Should not include non-GPT models"
        assert all(m.startswith(('gpt-3.5', 'gpt-4')) for m in models), "All models should be GPT-3.5 or GPT-4"


def test_list_available_models_api_error():
    """Test fallback behavior when API call fails."""
    with patch('openai.OpenAI') as mock_openai:
        # Simulate API error
        mock_openai.return_value.models.list.side_effect = OpenAIError("API Error")
        
        # Call the method
        models = OpenAIProvider.list_available_models()
        
        # Verify fallback to GPT_PRICING keys
        assert isinstance(models, list), "Should return a list"
        assert len(models) > 0, "Should return at least one model"
        assert "gpt-4" in models, "Should include models from GPT_PRICING"
        assert "gpt-3.5-turbo" in models, "Should include models from GPT_PRICING"


def test_list_available_models_unexpected_error():
    """Test error handling for unexpected exceptions."""
    with patch('openai.OpenAI') as mock_openai:
        # Simulate unexpected error
        mock_openai.return_value.models.list.side_effect = Exception("Unexpected error")
        
        # Verify error handling
        with pytest.raises(ProviderError) as exc_info:
            OpenAIProvider.list_available_models()
        
        assert "Error fetching available models" in str(exc_info.value)