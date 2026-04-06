"""
LLM client abstraction layer.
Supports Anthropic (Claude), OpenAI, and local Ollama for generation.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    text: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """Generate a response from the LLM."""
        pass


class OpenAIClient(LLMClient):
    """
    OpenAI API client for GPT models.

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        result = client.generate("Explain quantum computing.")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai required. Install: pip install openai")

        self.model = model
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return GenerationResult(
            text=response.choices[0].message.content,
            model=self.model,
            usage=usage,
        )


class OllamaClient(LLMClient):
    """
    Local LLM via Ollama (free, no API key needed).

    Requires Ollama running locally: https://ollama.ai

    Usage:
        client = OllamaClient(model="llama3.1:8b")
        result = client.generate("What is machine learning?")
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized Ollama client: {model} @ {base_url}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        # Estimate tokens from response
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return GenerationResult(
            text=data["response"],
            model=self.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )


class AnthropicClient(LLMClient):
    """
    Anthropic API client for Claude models.

    Usage:
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        result = client.generate("Explain quantum computing.")

    Note:
        Requires ANTHROPIC_API_KEY environment variable or explicit api_key.
        Anthropic's API differs from OpenAI in that:
        - system prompt is a top-level parameter, not a message role
        - max_tokens is required (not optional)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic required. Install: pip install anthropic")

        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        logger.info(f"Initialized Anthropic client with model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        # Extract text from content blocks
        text = "".join(block.text for block in response.content if block.type == "text")

        return GenerationResult(
            text=text,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            },
        )
