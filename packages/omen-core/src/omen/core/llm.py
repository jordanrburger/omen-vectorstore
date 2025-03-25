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
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not provided for {provider}")
            
        if self.provider == "openai":
            openai.api_key = self.api_key
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        self._setup_retry()
        
    def _setup_retry(self):
        """Set up retry decorator for API calls."""
        self.retry_decorator = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((openai.error.APIError, openai.error.RateLimitError)),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        
    @property
    def retry(self):
        """Get the retry decorator."""
        return self.retry_decorator
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
    @retry
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    @retry
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text using Anthropic API."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        return response.content[0].text
        
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured data from a prompt.
        
        Args:
            prompt: The prompt to generate from
            schema: JSON schema defining the structure
            system_prompt: Optional system prompt
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            Generated structured data
        """
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"""Generate JSON data matching this schema:
{schema_str}

Based on this prompt:
{prompt}

Return only valid JSON matching the schema."""
        
        response = self.generate(full_prompt, system_prompt, **kwargs)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in response")
            
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
            
    def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            model: Optional model to use for embedding
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            List of embedding values
        """
        if self.provider == "openai":
            return self._generate_embedding_openai(text, model, **kwargs)
        else:
            raise ValueError(f"Embedding not supported for provider: {self.provider}")
            
    @retry
    def _generate_embedding_openai(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """Generate embedding using OpenAI API."""
        model = model or "text-embedding-ada-002"
        
        response = openai.Embedding.create(
            model=model,
            input=text,
            **kwargs
        )
        
        return response.data[0].embedding 