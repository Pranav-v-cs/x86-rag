import json
import logging
import os
from typing import Optional

import requests

from src.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are an expert assistant focused solely on Intel x86-64 instruction set architecture.
Answer questions ONLY about x86-64 instruction semantics, opcode encoding, operands, flags affected,
exceptions, and CPUID/feature requirements. Do not answer questions about general OS development,
paging, protected mode, interrupts, or memory management unless they appear inline inside an
instruction's own description. If a question is outside this scope, politely decline to answer.

Always cite your sources using the format:
[Source: <filename> | Page <page> | <mnemonic>]

If multiple sources are used, list each one."""


class OllamaClient:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.base_url = OLLAMA_BASE_URL
        self._check_available()

    def _check_available(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = r.json().get("models", [])
            model_names = [m["name"] for m in models]
            if self.model not in model_names:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s. "
                    "Run: ollama pull %s",
                    self.model,
                    ", ".join(model_names),
                    self.model,
                )
        except requests.ConnectionError:
            logger.warning(
                "Cannot connect to Ollama at %s. "
                "Make sure Ollama is running (ollama serve).",
                self.base_url,
            )

    def generate(
        self, prompt: str, chunks: Optional[list[Chunk]] = None, temperature: float = 0.1
    ) -> str:
        context = ""
        if chunks:
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                page_str = f" | Page {chunk.page}" if chunk.page else ""
                context_parts.append(
                    f"[Source {i}]: {chunk.source}{page_str} | {chunk.mnemonic} | "
                    f"[{chunk.subsection}]\n{chunk.text}"
                )
            context = "\n\n".join(context_parts)

        user_prompt = f"""Using the provided reference sources, answer the following question concisely and accurately.

Question: {prompt}

Reference Sources:
{context}

Answer the question based solely on the provided sources. Include citations in the format:
[Source: <filename> | Page <page> | <mnemonic>]"""

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            response = r.json()
            return response.get("response", "").strip()
        except requests.ConnectionError:
            return (
                "Error: Cannot connect to Ollama. Ensure it's running "
                "(ollama serve) and the model is pulled (ollama pull <model>)."
            )
        except Exception as e:
            logger.error("Ollama request failed: %s", e)
            return f"Error generating response: {e}"
