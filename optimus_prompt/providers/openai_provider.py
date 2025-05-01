import time
from typing import Dict, Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from ..core.prompt import Prompt
from ..core.response import Response
from .base_provider import BaseProvider


# GPT model pricing per million tokens
GPT_PRICING = {
    "gpt-4": {
        "input_per_million": 30.0,    # $30 per million input tokens
        "output_per_million": 60.0,   # $60 per million output tokens
    },
    "gpt-4-32k": {
        "input_per_million": 60.0,    # $60 per million input tokens
        "output_per_million": 120.0,  # $120 per million output tokens
    },
    "gpt-3.5-turbo": {
        "input_per_million": 0.5,     # $0.50 per million input tokens
        "output_per_million": 1.5,    # $1.50 per million output tokens
    },
    "gpt-3.5-turbo-16k": {
        "input_per_million": 1.0,     # $1.00 per million input tokens
        "output_per_million": 2.0,    # $2.00 per million output tokens
    }
}


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI's API."""

    def __init__(self, model: str, api_key: str = None):
        """Initialize the OpenAI provider.

        Args:
            model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
        """
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)
        self._model_info = None

    def generate(self, prompt: Prompt) -> Response:
        """Generate a response using OpenAI's API.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            Response object containing the generated text and metadata

        Raises:
            ProviderError: If there is an error calling the OpenAI API
        """
        start_time = time.time()

        try:
            completion: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt.text}
                ]
            )
            
            end_time = time.time()
            latency = end_time - start_time

            response_text = completion.choices[0].message.content
            
            tokens = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens
            }
            
            cost = self.calculate_cost(tokens)
            
            metadata = {
                "latency": latency,
                "token_count": completion.usage.total_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "finish_reason": completion.choices[0].finish_reason,
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
            raise ProviderError(f"Error generating response from OpenAI: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dict containing model information
        """
        if self._model_info is None:
            # Cache the model info
            self._model_info = {
                "name": self.model,
                "provider": "openai",
                "capabilities": ["text", "code", "analysis"],
                # Model-specific max tokens
                "max_tokens": {
                    "gpt-4": 8192,
                    "gpt-4-32k": 32768,
                    "gpt-3.5-turbo": 4096,
                    "gpt-3.5-turbo-16k": 16384
                }.get(self.model, 4096)  # Default to 4096 if model not found
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
        pricing = GPT_PRICING.get(self.model, GPT_PRICING["gpt-3.5-turbo"])
        
        input_cost = (tokens["input_tokens"] / 1_000_000) * pricing["input_per_million"]
        output_cost = (tokens["output_tokens"] / 1_000_000) * pricing["output_per_million"]
        
        return round(input_cost + output_cost, 6)


class ProviderError(Exception):
    """Exception raised when there is an error with the provider."""
    pass