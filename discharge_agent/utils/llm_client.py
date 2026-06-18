from __future__ import annotations

import os
import time
from typing import List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import SETTINGS
from prompts.system_prompt import SYSTEM_PROMPT


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.model = model or SETTINGS.model
        if self.model.startswith("gemini-"):
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        self._client = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _get_client(self):
        if self._client is None:
            if self.model.startswith("gemini-1.5-flash"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self._client = genai
                except ImportError:
                    raise RuntimeError(
                        "google-generativeai package not installed. Run: pip install google-generativeai"
                    )
            else:
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    raise RuntimeError(
                        "anthropic package not installed. Run: pip install anthropic"
                    )
        return self._client

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(SETTINGS.max_retries_per_tool),
        wait=wait_exponential(
            multiplier=SETTINGS.retry_base_delay_s, min=1, max=10
        ),
        reraise=False,
    )
    def complete(
        self,
        user_prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Single-turn completion.
        Returns empty string on all failures (agent handles missing data).
        """
        client = self._get_client()
        system_text = system or SYSTEM_PROMPT
        max_tok = max_tokens or SETTINGS.max_tokens
        temp = temperature if temperature is not None else SETTINGS.temperature

        if self.model.startswith("gemini-"):
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            model = client.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tok,
                },
                system_instruction=system_text,
                safety_settings=safety_settings,
            )
            response = model.generate_content(user_prompt)
            try:
                text = response.text
                self.total_input_tokens += len(user_prompt.split())
                self.total_output_tokens += len(text.split())
            except Exception:
                text = ""
            return text

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tok,
            temperature=temp,
            system=system_text,
            messages=[{"role": "user", "content": user_prompt}],
        )

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response.content[0].text if response.content else ""

    def chat(
        self,
        messages: List[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Multi-turn chat completion for agent loop.
        messages: list of {"role": "user"|"assistant", "content": str}
        """
        client = self._get_client()
        system_text = system or SYSTEM_PROMPT
        max_tok = max_tokens or SETTINGS.max_tokens

        if self.model.startswith("gemini-"):
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            model = client.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": SETTINGS.temperature,
                    "max_output_tokens": max_tok,
                },
                system_instruction=system_text,
                safety_settings=safety_settings,
            )
            gemini_messages = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                gemini_messages.append({"role": role, "parts": [m["content"]]})
            response = model.generate_content(gemini_messages)
            try:
                text = response.text
                self.total_input_tokens += sum(len(m["content"].split()) for m in messages)
                self.total_output_tokens += len(text.split())
            except Exception:
                text = ""
            return text

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tok,
            temperature=SETTINGS.temperature,
            system=system_text,
            messages=messages,
        )

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response.content[0].text if response.content else ""

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
