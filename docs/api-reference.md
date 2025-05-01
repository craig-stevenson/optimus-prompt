# API Reference

## Core Module

### Prompt

```python
class Prompt:
    """Represents a prompt to be sent to LLM providers."""

    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new Prompt.

        Args:
            text: The prompt text
            metadata: Optional metadata dictionary
        """

    @classmethod
    def from_file(cls, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> 'Prompt':
        """
        Create a Prompt instance from a text file.

        Args:
            file_path: Path to the file containing the prompt text
            metadata: Optional metadata to attach to the prompt

        Returns:
            A new Prompt instance
        """

    def to_dict(self) -> Dict[str, Any]:
        """Convert the prompt to a dictionary representation."""
```

### ResponseWriter

```python
class ResponseWriter:
    """Handles writing prompts and responses to files."""

    def __init__(self, output_path: str):
        """
        Initialize a new ResponseWriter.

        Args:
            output_path: Path where responses will be written
        """

    def write(self, prompt: Prompt, responses: List[Response]):
        """
        Write prompt and responses to file.

        Args:
            prompt: The prompt that was used
            responses: List of responses from providers
        """
```

## Providers Module

### Base Provider

```python
class BaseProvider:
    """Base class for LLM providers."""

    def __init__(self, model: str):
        """
        Initialize a new provider.

        Args:
            model: Name of the model to use
        """

    def generate(self, prompt: Prompt) -> Response:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Response object containing the model's response
        """
```

### OpenAI Provider

```python
class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's GPT models."""

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize OpenAI provider.

        Args:
            model: Name of the OpenAI model to use (default: "gpt-4")
        """
```

### Anthropic Provider

```python
class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Claude models."""

    def __init__(self, model: str = "claude-3"):
        """
        Initialize Anthropic provider.

        Args:
            model: Name of the Anthropic model to use (default: "claude-3")
        """
```

## Response Class

```python
class Response:
    """Represents a response from an LLM provider."""

    def __init__(
        self,
        text: str,
        provider_name: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new Response.

        Args:
            text: The response text
            provider_name: Name of the provider (e.g., "OpenAI", "Anthropic")
            model: Name of the model used
            metadata: Optional metadata including tokens, latency, cost
        """

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary representation."""
```

## Metadata Structure

Responses include metadata with the following structure:

```python
metadata = {
    'token_count': int,  # Total tokens used
    'latency': float,    # Response time in seconds
    'cost': {
        'prompt_tokens': int,    # Tokens in the prompt
        'completion_tokens': int, # Tokens in the response
        'total_tokens': int,     # Total tokens used
        'prompt_cost': float,    # Cost for prompt tokens
        'completion_cost': float, # Cost for completion tokens
        'total': float          # Total cost in USD
    }
}