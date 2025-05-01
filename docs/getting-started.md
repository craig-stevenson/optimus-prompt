# Getting Started

This guide will help you get up and running with Optimus Prompt quickly.

## Installation

Install Optimus Prompt using pip:

```bash
pip install optimus-prompt
```

## Environment Setup

1. Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

2. Make sure you have Python 3.9 or later installed.

## Quick Start

Here's a minimal example to get you started:

```python
from optimus_prompt.core import Prompt
from optimus_prompt.providers import OpenAIProvider, AnthropicProvider
from optimus_prompt.core import ResponseWriter

# Create a prompt
prompt = Prompt("Explain quantum computing in simple terms")

# Initialize providers
providers = [
    OpenAIProvider(model="gpt-4"),
    AnthropicProvider(model="claude-3")
]

# Collect responses
responses = []
for provider in providers:
    response = provider.generate(prompt)
    responses.append(response)

# Write responses to file
writer = ResponseWriter("output.txt")
writer.write(prompt, responses)
```

### Using a Prompt File

For longer prompts, you can use a text file:

1. Create `prompt.txt`:
```text
Explain quantum computing in simple terms
```

2. Load the prompt from file:
```python
prompt = Prompt.from_file("prompt.txt")
```

## Next Steps

- Check out the [Usage Guide](./usage-guide.md) for more detailed examples
- See the [API Reference](./api-reference.md) for complete documentation