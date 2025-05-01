import os
from dotenv import load_dotenv

from optimus_prompt.core import Prompt
from optimus_prompt.providers import OpenAIProvider, AnthropicProvider
from optimus_prompt.core import ResponseWriter

# Load environment variables from .env file
load_dotenv()

def main():
    # Create a prompt from a file
    prompt = Prompt.from_file("prompt.txt")

    # Initialize providers with API keys from environment variables
    p1 = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    p2 = OpenAIProvider(model="gpt-3.5-turbo")
    providers = [p1, p2]

    # Collect responses from all providers
    responses = []
    for provider in providers:
        try:
            print(f"Getting response from {provider.name} ({provider.model})...")
            response = provider.generate(prompt)
            responses.append(response)
            print(f"✓ Response received from {provider.name}")
        except Exception as e:
            print(f"✗ Error from {provider.name}: {str(e)}")

    # Write responses to file
    writer = ResponseWriter("responses.txt")
    writer.write(prompt, responses)
    print(f"\nResponses have been written to responses.txt")

    # Print some basic statistics
    print("\nResponse Statistics:")
    total_cost = 0.0
    for response in responses:
        latency = response.metadata.get('latency', 'N/A')
        tokens = response.metadata.get('token_count', 'N/A')
        cost = response.metadata.get('cost', {}).get('total', 'N/A')
        
        if isinstance(cost, (int, float)):
            total_cost += cost
        
        print(f"{response.provider_name} ({response.model}):")
        print(f"  - Latency: {latency:.2f}s" if isinstance(latency, float) else f"  - Latency: {latency}")
        print(f"  - Tokens: {tokens}")
        print(f"  - Cost: ${cost:.4f}" if isinstance(cost, (int, float)) else f"  - Cost: {cost}")
    
    if total_cost > 0:
        print(f"\nTotal cost: ${total_cost:.4f}")

if __name__ == "__main__":
    # Create example prompt.txt if it doesn't exist
    if not os.path.exists("prompt.txt"):
        with open("prompt.txt", "w") as f:
            f.write("Explain quantum computing in simple terms.")
        print("Created example prompt.txt")

    # Run the main function
    main()