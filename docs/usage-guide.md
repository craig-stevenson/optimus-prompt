# Usage Guide

This guide covers the main features and common use cases of Optimus Prompt.

## Basic Usage

### Creating Prompts

You can create prompts directly or load them from a file:

```python
# Direct creation
prompt = Prompt("What is artificial intelligence?")

# From file
prompt = Prompt.from_file("prompt.txt")
```

### Working with Providers

Optimus Prompt supports multiple LLM providers. Each provider requires appropriate API keys in your `.env` file.

```python
# Initialize providers with specific models
openai = OpenAIProvider(model="gpt-4")
anthropic = AnthropicProvider(model="claude-3")

# Create a list of providers
providers = [openai, anthropic]
```

### Collecting Responses

```python
responses = []
for provider in providers:
    try:
        print(f"Getting response from {provider.name}...")
        response = provider.generate(prompt)
        responses.append(response)
        print(f"✓ Response received")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
```

## Response Analysis

### Accessing Response Data

Each response object contains:
- The response text
- Provider information
- Metadata (tokens, latency, cost)

```python
for response in responses:
    print(f"\n{response.provider_name} ({response.model}):")
    print(f"Response: {response.text}")
    print(f"Tokens: {response.metadata['token_count']}")
    print(f"Latency: {response.metadata['latency']:.2f}s")
    print(f"Cost: ${response.metadata['cost']['total']:.4f}")
```

### Writing Responses

The ResponseWriter class helps format and save responses:

```python
writer = ResponseWriter("responses.txt")
writer.write(prompt, responses)
```

Output format:
```
Prompt: What is artificial intelligence?
Time: 2024-04-30 22:27:00

=== GPT-4 ===
[Response text here]
Tokens: 150
Latency: 2.5s
Cost: $0.0030

=== Claude ===
[Response text here]
Tokens: 180
Latency: 1.8s
Cost: $0.0025
```

## Cost Tracking

Optimus Prompt automatically tracks costs for each provider:

```python
total_cost = 0.0
for response in responses:
    cost = response.metadata.get('cost', {}).get('total', 0)
    total_cost += cost
    
print(f"\nTotal cost: ${total_cost:.4f}")
```

## Error Handling

Always wrap provider calls in try-except blocks to handle potential API errors:

```python
try:
    response = provider.generate(prompt)
except Exception as e:
    print(f"Error with {provider.name}: {str(e)}")
    # Handle error appropriately
```

Common errors include:
- Invalid API keys
- Rate limiting
- Network issues
- Invalid model names

## Best Practices

1. **API Keys**: Always use environment variables for API keys
2. **Error Handling**: Implement proper error handling for API calls
3. **Cost Management**: Monitor and track API costs
4. **Response Storage**: Save responses for later analysis
5. **Model Selection**: Choose appropriate models for your use case