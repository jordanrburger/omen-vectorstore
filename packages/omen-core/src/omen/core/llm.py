"""
LLM client for interacting with language models.

This module provides a client for generating text using various language models,
with support for different providers and models.
"""

import json
import logging
import os
import time
import re
from typing import Dict, List, Any, Optional, Union, Tuple

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Add Anthropic import
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with language models."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 6000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the LLM client.
        
        Args:
            provider: The LLM provider to use ("openai" or "anthropic")
            model: The model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider.lower()
        if self.provider not in ["openai", "anthropic"]:
            raise ValueError("Invalid provider. Must be 'openai' or 'anthropic'")
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not provided for {provider}")
        
        if self.provider == "openai":
            self.client = openai.Client(api_key=self.api_key)
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Invalid provider: {provider}")
        
        self._setup_retry()

    def _setup_retry(self):
        """Set up retry decorator for API calls."""
        self.retry_decorator = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((Exception,)),  # Retry on any exception
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )

    def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI."""
        @self.retry_decorator
        def _call_openai():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        return _call_openai()

    def _generate_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Anthropic."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed")
        
        @self.retry_decorator
        def _call_anthropic():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.content[0].text
        
        return _call_anthropic()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using the configured provider.
        
        Args:
            prompt: The prompt to generate text from
            system_prompt: Optional system prompt to set context
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt)
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of embedding values
        """
        if self.provider == "openai":
            @self.retry_decorator
            def _call_openai():
                response = self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text
                )
                return response.data[0].embedding
            
            return _call_openai()
        else:
            raise ValueError(f"Embeddings not supported for provider: {self.provider}")

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured data using the configured provider.
        
        Args:
            prompt: The prompt to generate data from
            schema: JSON schema describing the expected structure
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated structured data
        """
        # Add schema to prompt
        schema_prompt = f"""
        Generate a JSON object matching this schema:
        {json.dumps(schema, indent=2)}
        
        Return only the JSON object, no other text.
        """
        
        # Generate response
        response = self.generate(prompt, system_prompt=schema_prompt)
        
        # Extract JSON from response
        try:
            # Find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")
            
            # Parse JSON
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}") 