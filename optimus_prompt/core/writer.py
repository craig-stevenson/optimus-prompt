from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO

from .prompt import Prompt
from .response import Response


class ResponseWriter:
    """Handles writing responses to files in a human-readable format."""

    def __init__(self, output_path: str):
        """Initialize the writer with an output path.

        Args:
            output_path: Path where responses will be written
        """
        self.output_path = Path(output_path)

    def write(self, prompt: Prompt, responses: List[Response]) -> None:
        """Write prompt and responses to the output file.

        Args:
            prompt: The prompt that was used
            responses: List of responses from different providers
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            self._write_header(f)
            self._write_prompt(f, prompt)
            self._write_responses(f, responses)
            self._write_footer(f)

    def _write_header(self, f: TextIO) -> None:
        """Write the header section of the output file.

        Args:
            f: The file object to write to
        """
        f.write("=" * 80 + "\n")
        f.write(f"Prompt Prism Results - {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

    def _write_prompt(self, f: TextIO, prompt: Prompt) -> None:
        """Write the prompt section of the output file.

        Args:
            f: The file object to write to
            prompt: The prompt to write
        """
        f.write("PROMPT:\n")
        f.write("-" * 80 + "\n")
        f.write(prompt.text + "\n")
        if prompt.metadata:
            f.write("\nPrompt Metadata:\n")
            for key, value in prompt.metadata.items():
                f.write(f"- {key}: {value}\n")
        f.write("\n" + "=" * 80 + "\n\n")

    def _write_responses(self, f: TextIO, responses: List[Response]) -> None:
        """Write all responses to the output file.

        Args:
            f: The file object to write to
            responses: List of responses to write
        """
        for i, response in enumerate(responses, 1):
            f.write(f"RESPONSE #{i}:\n")
            f.write("-" * 80 + "\n")
            f.write(response.format_for_output())
            f.write("\n\n" + "-" * 80 + "\n\n")

    def _write_footer(self, f: TextIO) -> None:
        """Write the footer section of the output file.

        Args:
            f: The file object to write to
        """
        f.write("=" * 80 + "\n")
        f.write("End of Results\n")
        f.write("=" * 80 + "\n")

    def append_response(self, response: Response) -> None:
        """Append a single response to an existing output file.

        Args:
            response: The response to append
        """
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"RESPONSE #{self._count_existing_responses() + 1}:\n")
            f.write("-" * 80 + "\n")
            f.write(response.format_for_output())
            f.write("\n\n" + "-" * 80 + "\n")

    def _count_existing_responses(self) -> int:
        """Count the number of responses in the existing output file.

        Returns:
            Number of responses found in the file
        """
        if not self.output_path.exists():
            return 0

        with open(self.output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count("RESPONSE #")