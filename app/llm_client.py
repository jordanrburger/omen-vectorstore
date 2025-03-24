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

class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMResponseError(LLMError):
    """Exception for invalid LLM responses."""
    pass

class LLMClient:
    """Client for interacting with language models."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: The LLM provider (openai, anthropic, etc.)
            model: The model to use (defaults to recommended model for provider)
            api_key: API key for the provider (defaults to environment variable)
            temperature: Temperature for generation (0.0-1.0)
            max_tokens: Maximum tokens for generation
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries in seconds
            timeout: Request timeout in seconds
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.client = None
        
        # Set default model based on provider
        if model is None:
            if self.provider == "openai":
                self.model = "gpt-4"
            elif self.provider == "anthropic":
                self.model = "claude-3-opus-20240229"  # o1 model
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
            elif self.provider == "anthropic":
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key is None:
                    raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY not set")
            else:
                raise ValueError(f"API key not provided for provider: {provider}")
        
        # Initialize provider-specific client
        if self.provider == "openai":
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=self.timeout
            )
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
            self.client = anthropic.Anthropic(
                api_key=api_key,
                timeout=self.timeout
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM client with provider={provider}, model={self.model}")
    
    def _validate_response(self, response: Any, expected_format: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the LLM response.
        
        Args:
            response: The response to validate
            expected_format: Expected format of the response
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if expected_format == "json":
            if isinstance(response, str):
                try:
                    json.loads(response)
                    return True, None
                except json.JSONDecodeError:
                    return False, "Invalid JSON string"
            elif isinstance(response, (dict, list)):
                return True, None
            else:
                return False, f"Unexpected response type: {type(response)}"
        else:
            if isinstance(response, str):
                return True, None
            else:
                return False, f"Unexpected response type: {type(response)}"
    
    def _extract_json_from_text(self, text: str) -> Optional[Union[Dict, List]]:
        """
        Extract JSON from text that might contain other content.
        
        Args:
            text: Text to extract JSON from
            
        Returns:
            Extracted JSON or None if not found
        """
        # First try to parse the entire string as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the text (between curly braces)
        try:
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try to find JSON array in the text (between square brackets)
        try:
            json_match = re.search(r'(\[.*\])', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try extracting code blocks that might contain JSON
        try:
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
            for block in code_blocks:
                try:
                    return json.loads(block.strip())
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, expected_format: str = "text") -> str:
        """Generate text using the configured LLM.
        
        Args:
            prompt: The prompt to generate text from
            system_prompt: Optional system prompt to set context
            expected_format: Expected format of response ("text" or "json")
            
        Returns:
            Generated text
            
        Raises:
            Exception: If all LLM call attempts fail
        """
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        
        if expected_format == "json":
            system_prompt += " Provide your response as valid JSON."
        
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, system_prompt, expected_format)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, expected_format)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise Exception(f"All LLM call attempts failed: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _generate_openai(self, prompt: str, system_prompt: str, expected_format: str) -> str:
        """Generate text using OpenAI API.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: System prompt for setting context
            expected_format: Expected format of response ("text" or "json")
            
        Returns:
            Generated text
        """
        if not self.client:
            raise ValueError("OpenAI client not set")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Models that support JSON response format
        json_supported_models = ["gpt-4-turbo", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-3.5-turbo-1106", "gpt-3.5-turbo"]
        
        # Prepare API request
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add JSON response format for supported models
        if expected_format == "json" and self.model in json_supported_models:
            logger.info(f"Using JSON response format with model {self.model}")
            request_params["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**request_params)
        
        # Extract content from response
        response_text = response.choices[0].message.content.strip()
        
        # Process response based on expected format
        if expected_format == "json":
            json_response = self._extract_json_from_text(response_text)
            if json_response:
                return json_response
            # Fall back to returning the text if JSON extraction fails
            return response_text
        
        return response_text
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _generate_anthropic(self, prompt: str, system_prompt: str, expected_format: str) -> str:
        """Generate text using Anthropic API.
        
        Args:
            prompt: The prompt to generate from
            system_prompt: System prompt for setting context
            expected_format: Expected format of response ("text" or "json")
            
        Returns:
            Generated text
        """
        if not self.client:
            raise ValueError("Anthropic client not set")
        
        # Map model names to actual Anthropic model IDs if needed
        model_map = {
            "o1": "claude-3-opus-20240229",
            "o3-mini": "claude-3-haiku-20240307",
            "claude-opus": "claude-3-opus-20240229",
            "claude-opus-mini": "claude-3-haiku-20240307"
        }
        
        model = model_map.get(self.model, self.model)
        
        # Add JSON structure instruction for expected JSON format
        if expected_format == "json":
            prompt += "\n\nRespond only with a valid JSON object, with no additional text before or after."
        
        # Create the message
        response = self.client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract content from response
        response_text = response.content[0].text
        
        # Process response based on expected format
        if expected_format == "json":
            json_response = self._extract_json_from_text(response_text)
            if json_response:
                return json_response
            # Fall back to returning the text if JSON extraction fails
            return response_text
        
        return response_text 