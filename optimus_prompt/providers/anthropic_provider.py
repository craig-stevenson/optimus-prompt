import time
from typing import Dict, Any

from anthropic import Anthropic

from ..core.prompt import Prompt
from ..core.response import Response
from .base_provider import BaseProvider


# Claude model pricing per million tokens
CLAUDE_PRICING = {
    "claude-3-5-opus-20240620": {
        "input_per_million": 15.0,   # $15 per million input tokens
        "output_per_million": 75.0,  # $75 per million output tokens
    },
    "claude-3-5-sonnet-20240620": {
        "input_per_million": 3.0,    # $3 per million input tokens
        "output_per_million": 15.0,  # $15 per million output tokens
    },
    "claude-3-5-haiku-20240620": {
        "input_per_million": 0.25,   # $0.25 per million input tokens
        "output_per_million": 1.25,  # $1.25 per million output tokens
    }
}


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic's API."""

    def __init__(self, model: str, api_key: str = None):
        """Initialize the Anthropic provider.

        Args:
            model: The model to use (e.g., "claude-3-5-opus-20240620", "claude-3-5-sonnet-20240620")
            api_key: Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var
        """
        super().__init__(model)
        self.client = Anthropic(api_key=api_key)
        self._model_info = None

    def generate(self, prompt: Prompt) -> Response:
        """Generate a response using Anthropic's API.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            Response object containing the generated text and metadata

        Raises:
            ProviderError: If there is an error calling the Anthropic API
        """
        start_time = time.time()

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.text
                    }
                ]
            )
            
            end_time = time.time()
            latency = end_time - start_time

            response_text = message.content[0].text
            
            tokens = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens
            }
            
            cost = self.calculate_cost(tokens)
            
            metadata = {
                "latency": latency,
                "stop_reason": message.stop_reason,
                "message_id": message.id,
                "token_count": message.usage.input_tokens + message.usage.output_tokens,
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "cost": {
                    "total": cost,
                    "input_tokens": tokens["input_tokens"],
                    "output_tokens": tokens["output_tokens"]
                }
            }

            return Response(
                text=response_text,
                prompt=prompt,
                provider_name=self.name,
                model=self.model,
                metadata=metadata
            )

        except Exception as e:
            raise ProviderError(f"Error generating response from Anthropic: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dict containing model information
        """
        if self._model_info is None:
            pricing = CLAUDE_PRICING.get(self.model, CLAUDE_PRICING["claude-3-5-sonnet-20240620"])
            
            # Cache the model info
            self._model_info = {
                "name": self.model,
                "provider": "anthropic",
                "capabilities": ["text", "code", "analysis"],
                # Model-specific max tokens
                "max_tokens": {
                    "claude-3-5-opus-20240620": 32768,
                    "claude-3-5-sonnet-20240620": 32768,
                    "claude-3-5-haiku-20240620": 32768,
                }.get(self.model, 32768),  # Default to 32k if model not found
                "pricing": pricing
            }
        return self._model_info

    def calculate_cost(self, tokens: Dict[str, int]) -> float:
        """Calculate the cost of the API call based on token usage.

        Args:
            tokens: Dictionary containing token counts with keys:
                   - input_tokens: Number of tokens in the prompt
                   - output_tokens: Number of tokens in the completion

        Returns:
            float: The total cost in USD
        """
        pricing = CLAUDE_PRICING.get(self.model, CLAUDE_PRICING["claude-3-5-sonnet-20240620"])
        
        input_cost = (tokens["input_tokens"] / 1_000_000) * pricing["input_per_million"]
        output_cost = (tokens["output_tokens"] / 1_000_000) * pricing["output_per_million"]
        
        return round(input_cost + output_cost, 6)


class ProviderError(Exception):
    """Exception raised when there is an error with the provider."""
    pass