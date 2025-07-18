import requests
import os
import anthropic
from typing import Optional

class LLMService:
    """Unified service for interacting with various LLM APIs (GPT, LLaMA, Anthropic, or others)."""
    
    def __init__(self, 
                 provider: str,        # e.g. "openai" or "llama" or "anthropic"
                 api_url: str, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        """
        Initialize the LLM service.

        Args:
            provider (str): The name of the provider (e.g., "openai", "llama", "anthropic").
            api_url (str): The URL of the LLM API for the given provider.
            api_key (str, optional): API key for authentication.
            model (str, optional): Specific model to use.
        """
        self.provider = provider.lower().strip()
        self.api_url = api_url
        
        # Try to get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY")
        self.model = model
        
        # Basic validation
        if not self.provider:
            raise ValueError("Provider must be specified (e.g. 'openai', 'llama', 'anthropic').")
        if not self.api_url:
            raise ValueError("API URL must be provided.")
        if not self.api_key:
            raise ValueError(f"API key must be provided as parameter or set in {self.provider.upper()}_API_KEY environment variable.")

    def call_llm(self, 
                 prompt: str, 
                 max_tokens: int = 4000, 
                 temperature: float = 0.3) -> str:
        """
        Make a call to the LLM API, branching on the provider.

        Args:
            prompt (str): The prompt text to send to the LLM.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature.

        Returns:
            str: The text response from the LLM.
            
        Raises:
            ValueError: If there's an issue with the API response format.
            requests.RequestException: If there's an issue with the HTTP request.
        """
        # Common headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # import pdb; pdb.set_trace()
        # Branch on provider
        if self.provider == "openai":
            # Example: GPT via OpenAI Chat Completions
            data = {
                "model": self.model or "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert at solving assigned tasks."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract text with better error handling
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Unexpected OpenAI API response format: missing 'choices'")
            if "message" not in response_data["choices"][0]:
                raise ValueError("Unexpected OpenAI API response format: missing 'message'")
            if "content" not in response_data["choices"][0]["message"]:
                raise ValueError("Unexpected OpenAI API response format: missing 'content'")
                
            return response_data["choices"][0]["message"]["content"]

        elif self.provider == "llama":
            # Example: LLaMA endpoint
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert at solving assigned tasks."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated subqueries from the response
            content = response_data["choices"][0]["message"]["content"]
            return content
            
        elif self.provider == "anthropic":
            # Anthropic Claude API
            try:
                import anthropic
            except ImportError:
                raise ImportError("The 'anthropic' package is required but not installed. Install it with: pip install anthropic")
                
            # Initialize Anthropic client
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Create the message
            response = client.messages.create(
                model=self.model or "claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                system="You are an expert at solving assigned tasks.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature
            )
            
            # Return the content from the response
            return response.content[0].text

        else:
            # Extend here for other providers
            raise NotImplementedError(f"Provider '{self.provider}' not implemented.")

