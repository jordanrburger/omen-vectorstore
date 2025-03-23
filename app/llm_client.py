"""
LLM client for interacting with language models.

This module provides a client for generating text using various language models,
with support for different providers and models.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with language models."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: The LLM provider (openai, huggingface, etc.)
            model: The model to use (defaults to recommended model for provider)
            api_key: API key for the provider (defaults to environment variable)
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens for generation
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default model based on provider
        if model is None:
            if self.provider == "openai":
                self.model = "gpt-4"
            else:
                raise ValueError(f"No default model for provider: {provider}")
        else:
            self.model = model
        
        # Set API key from environment variable if not provided
        if api_key is None:
            if self.provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
            else:
                raise ValueError(f"API key not provided for provider: {provider}")
        
        # Initialize provider-specific client
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM client with provider={provider}, model={self.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError))
    )
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str,
        expected_format: str = "text"
    ) -> Union[str, Dict, List]:
        """
        Generate text using the language model.
        
        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            expected_format: Expected format of the response (text, json, etc.)
            
        Returns:
            Generated text or parsed JSON if expected_format is "json"
        """
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt, expected_format)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(
        self, 
        system_prompt: str, 
        user_prompt: str,
        expected_format: str = "text"
    ) -> Union[str, Dict, List]:
        """
        Generate text using the OpenAI API.
        
        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model
            expected_format: Expected format of the response (text, json, etc.)
            
        Returns:
            Generated text or parsed JSON if expected_format is "json"
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add JSON mode if expected format is json
        response_format = {"type": "json_object"} if expected_format == "json" else None
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON if expected_format is json
        if expected_format == "json" and content:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Try to extract JSON from the content
                import re
                json_match = re.search(r'(\{|\[).*(\}|\])', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.error("Failed to extract JSON from response")
                
                # Return the raw content if JSON parsing fails
                return content
        
        return content 