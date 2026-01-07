from __future__ import annotations

import httpx
import time
from dataclasses import dataclass
from typing import Optional

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str | None = None


class LLMClient:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def chat(self, model: str, system: str, user: str) -> LLMResponse:
        start = time.time()
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.7,
                },
            )
            latency_ms = int((time.time() - start) * 1000)

            if response.status_code != 200:
                return LLMResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

            data = response.json()

            # Handle OpenRouter error responses
            if "error" in data:
                return LLMResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error=data["error"].get("message", str(data["error"])),
                )

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException:
            latency_ms = int((time.time() - start) * 1000)
            return LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                error="Request timed out",
            )
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            return LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                error=str(e),
            )

    def close(self):
        self.client.close()
