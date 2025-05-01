# Optimus Prompt 🤖

Compare responses from different Language Learning Models (LLMs) to the same prompt.

## Features

- Support for multiple LLM providers:
  - OpenAI (GPT-3.5-turbo, GPT-4)
  - Anthropic (Claude-3)
- Simple prompt input via text file
- Detailed response statistics (latency, token count, cost)
- Human-readable output format
- Comprehensive error handling
- Easy to extend with new providers
- Built with type safety (mypy)

## Requirements

- Python 3.9 or higher
- Valid API keys for the LLM providers you want to use

## Installation

```bash
pip install optimus-prompt
```

## Quick Start

1. Create a `prompt.txt` file with your prompt:
```text
Explain quantum computing in simple terms
```

2. Set up your API keys in a `.env` file:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

3. Run Optimus Prompt:
```python
from dotenv import load_dotenv
from optimus_prompt.core import Prompt
from optimus_prompt.providers import OpenAIProvider, AnthropicProvider
from optimus_prompt.core import ResponseWriter

# Load environment variables
load_dotenv()

# Create a prompt from file
prompt = Prompt.from_file("prompt.txt")

# Initialize providers
providers = [
    AnthropicProvider(model="claude-3-5-sonnet-20240620"),
    OpenAIProvider(model="gpt-3.5-turbo")
]

# Collect responses
responses = []
for provider in providers:
    response = provider.generate(prompt)
    responses.append(response)

# Write responses to file
writer = ResponseWriter("responses.txt")
writer.write(prompt, responses)
```

For more examples, check out the [examples directory](examples/).

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Usage Guide](docs/usage-guide.md)
- [API Reference](docs/api-reference.md)

## Project Structure

```
optimus_prompt/
├── docs/                 # Documentation
├── examples/            # Example scripts and files
├── optimus_prompt/      # Main package
│   ├── core/           # Core functionality
│   ├── providers/      # LLM provider implementations
│   └── output/         # Output formatting
└── tests/              # Test suite
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/optimus-prompt.git
cd optimus-prompt
```

2. Install Poetry (if not already installed):
```bash
pip install poetry
```

3. Install dependencies:
```bash
poetry install
```

### Development Tools

The project uses several development tools to ensure code quality:

- `black`: Code formatting
- `isort`: Import sorting
- `mypy`: Static type checking
- `ruff`: Fast Python linter
- `pytest`: Testing framework

Run the tools:

```bash
# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy .

# Linting
poetry run ruff .

# Run tests
poetry run pytest
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.