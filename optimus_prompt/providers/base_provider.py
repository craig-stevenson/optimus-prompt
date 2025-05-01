from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from ..core.prompt import Prompt
from ..core.response import Response


class BaseProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model: str):
        """Initialize the provider with a specific model.

        Args:
            model: The model identifier to use for this provider
        """
        self.model = model

    @abstractmethod
    def generate(self, prompt: Prompt) -> Response:
        """Generate a response for the given prompt.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            Response: The generated response

        Raises:
            ProviderError: If there is an error generating the response
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dict containing model information like:
            {
                "name": "gpt-4",
                "provider": "openai",
                "max_tokens": 8192,
                "capabilities": ["text", "code", "analysis"]
            }
        """
        pass

    @abstractmethod
    def calculate_cost(self, tokens: Dict[str, int]) -> float:
        """Calculate the cost of the API call based on token usage.

        Args:
            tokens: Dictionary containing token counts with keys:
                   - input_tokens: Number of tokens in the prompt
                   - output_tokens: Number of tokens in the completion

        Returns:
            float: The total cost in USD
        """
        pass

    @property
    def name(self) -> str:
        """Get the provider name.

        Returns:
            The name of this provider (e.g. "openai", "anthropic")
        """
        return self.__class__.__name__.lower().replace("provider", "")