"""Phrasing LLM — converts structured facts into natural-language answers.

Loads a small local instruct model (Phi-3-mini-4k-instruct by default) and
generates text that ONLY phrases the given facts.  The system prompt is
**load-bearing** for the project's no-hallucination guarantee — do not relax
it for "better sounding" prose.

Public API
----------
phrase(facts) -> str
PhrasingLLM         — class for explicit lifecycle management
"""

from __future__ import annotations

import json
import logging
from typing import Any

from satquery.utils.config import DEVICE, PHRASING_MODEL_ID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — load-bearing, non-negotiable
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a factual report writer for satellite image analysis. "
    "Convert the structured facts below into a clear, professional "
    "natural-language answer.\n\n"
    "RULES (mandatory):\n"
    "1. ONLY phrase the given facts. Do NOT add, infer, or hallucinate "
    "any information not present in the input.\n"
    "2. If a field is missing or null, say so explicitly — do NOT guess.\n"
    "3. Use precise language. Prefer specific numbers over vague qualifiers.\n"
    "4. Keep the answer concise — one to three paragraphs maximum.\n"
    "5. Do NOT speculate about causes, trends, or context beyond what the "
    "facts state."
)


class PhrasingLLM:
    """Lazy-loaded phrasing model wrapper.

    The model is not loaded until the first call to ``phrase()``, so
    importing this module is cheap.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self._model_id = model_id or PHRASING_MODEL_ID
        self._device = device or DEVICE
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Load model and tokenizer on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading phrasing model: %s", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map=self._device,
            trust_remote_code=True,
        )

    def phrase(self, facts: dict[str, Any]) -> str:
        """Convert structured facts to natural language.

        Parameters
        ----------
        facts : dict[str, Any]
            Structured analysis results from specialist + verifier.
            Must be JSON-serializable.

        Returns
        -------
        str
            A natural-language answer grounded solely in *facts*.
        """
        self._load()

        facts_json = json.dumps(facts, indent=2, default=str)

        user_msg = f"Structured analysis facts:\n```json\n{facts_json}\n```"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Use chat template if available
        try:
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            input_text = f"{SYSTEM_PROMPT}\n\n{user_msg}\n\nAnswer:"

        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )

        answer = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return answer


# ---------------------------------------------------------------------------
# Module-level convenience function (uses a shared singleton)
# ---------------------------------------------------------------------------

_default_instance: PhrasingLLM | None = None


def phrase(facts: dict[str, Any]) -> str:
    """Convenience wrapper around ``PhrasingLLM.phrase()``.

    Uses a module-level singleton so the model is loaded only once.
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = PhrasingLLM()
    return _default_instance.phrase(facts)
