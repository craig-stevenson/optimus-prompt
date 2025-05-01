from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Prompt:
    """Represents a prompt to be sent to LLM providers."""

    text: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.now()

    def __post_init__(self):
        """Validate and process the prompt after initialization."""
        if not self.text.strip():
            raise ValueError("Prompt text cannot be empty")
        
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_file(cls, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> 'Prompt':
        """Create a Prompt instance from a text file.

        Args:
            file_path: Path to the file containing the prompt text
            metadata: Optional metadata to attach to the prompt

        Returns:
            A new Prompt instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            return cls(text=text, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the prompt to a dictionary representation.

        Returns:
            Dict containing the prompt data
        """
        return {
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

    def __str__(self) -> str:
        """Get string representation of the prompt.

        Returns:
            The prompt text
        """
        return self.text

    def __len__(self) -> int:
        """Get the length of the prompt text.

        Returns:
            Number of characters in the prompt text
        """
        return len(self.text)