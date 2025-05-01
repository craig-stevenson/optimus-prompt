from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from .prompt import Prompt


@dataclass
class Response:
    """Represents a response from an LLM provider."""

    text: str
    prompt: Prompt
    provider_name: str
    model: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.now()

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def latency(self) -> Optional[float]:
        """Get the response latency if available.

        Returns:
            Response time in seconds, or None if not available
        """
        return self.metadata.get('latency')

    @property
    def token_count(self) -> Optional[int]:
        """Get the response token count if available.

        Returns:
            Number of tokens in the response, or None if not available
        """
        return self.metadata.get('token_count')

    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary representation.

        Returns:
            Dict containing the response data
        """
        return {
            'text': self.text,
            'prompt': self.prompt.to_dict(),
            'provider_name': self.provider_name,
            'model': self.model,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

    def format_for_output(self) -> str:
        """Format the response for human-readable output.

        Returns:
            Formatted string containing response details
        """
        output = [
            f"Provider: {self.provider_name}",
            f"Model: {self.model}",
            f"Timestamp: {self.created_at.isoformat()}",
            "",
            "Response:",
            self.text,
            "",
            "Metadata:"
        ]

        # Add metadata if available
        if self.latency is not None:
            output.append(f"- Latency: {self.latency:.2f}s")
        if self.token_count is not None:
            output.append(f"- Token count: {self.token_count}")
        
        # Add any additional metadata
        for key, value in self.metadata.items():
            if key not in ('latency', 'token_count'):
                output.append(f"- {key}: {value}")

        return "\n".join(output)

    def __str__(self) -> str:
        """Get string representation of the response.

        Returns:
            The response text
        """
        return self.text

    def __len__(self) -> int:
        """Get the length of the response text.

        Returns:
            Number of characters in the response text
        """
        return len(self.text)