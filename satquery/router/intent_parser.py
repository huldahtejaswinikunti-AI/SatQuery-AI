"""LLM-backed intent parser — fallback for low-confidence router results.

Called ONLY when ``task_router.route()`` returns a confidence below
``ROUTER_CONFIDENCE_THRESHOLD``.  Must still select from the fixed
``TaskType`` enum — it never free-forms a new task name.

This is intentionally a thin wrapper; the rule-based router does the
real work.

Public API
----------
parse_intent(query, input_config) -> TaskType
"""

from __future__ import annotations

import json
import logging
from typing import Any

from satquery.router.task_types import TaskType
from satquery.utils.config import DEVICE, INTENT_PARSER_MODEL_ID

logger = logging.getLogger(__name__)

# The valid task names the model is allowed to produce.
_VALID_TASK_NAMES: list[str] = [t.value for t in TaskType]

_SYSTEM_PROMPT = f"""\
You are a task classifier for a satellite image analysis system.
Given a user query and image configuration, select EXACTLY ONE task type
from this fixed list:

{json.dumps(_VALID_TASK_NAMES, indent=2)}

Rules:
- single_vqa: single image + factual question
- single_caption: single image + request to describe/summarise
- grounding: request to locate/highlight objects in an image
- change_vqa: two images from different times + question about changes
- optical_sar_fusion: one optical + one SAR image to fuse/compare

Respond with ONLY the task type string, nothing else.
"""


class _ModelHolder:
    """Lazy singleton so the model is loaded only when actually needed."""

    _instance: _ModelHolder | None = None
    _model = None
    _tokenizer = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading intent parser model: %s", INTENT_PARSER_MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(
            INTENT_PARSER_MODEL_ID,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            INTENT_PARSER_MODEL_ID,
            device_map=DEVICE,
            trust_remote_code=True,
        )

    @property
    def model(self):
        self.load()
        return self._model

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer


def parse_intent(
    query: str,
    input_config: dict[str, Any],
) -> TaskType:
    """Classify *query* into a ``TaskType`` using a local instruct model.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    input_config : dict
        Same shape as ``task_router.route``'s *input_config*.

    Returns
    -------
    TaskType
        The best-matching task type from the fixed enum.
    """
    holder = _ModelHolder.get()

    user_msg = (
        f"Query: {query}\n"
        f"Image count: {input_config.get('image_count', 1)}\n"
        f"Modalities: {input_config.get('modalities', [])}\n"
        f"\nTask type:"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Tokenize using chat template if available, else simple concat
    try:
        input_text = holder.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        input_text = f"{_SYSTEM_PROMPT}\n\n{user_msg}"

    inputs = holder.tokenizer(input_text, return_tensors="pt").to(DEVICE)

    outputs = holder.model.generate(
        **inputs,
        max_new_tokens=32,
        temperature=0.1,
        do_sample=False,
    )

    raw = holder.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip().lower()

    return _fuzzy_match(raw)


def _fuzzy_match(raw: str) -> TaskType:
    """Map raw model output to the closest ``TaskType`` value.

    Falls back to ``SINGLE_VQA`` if nothing matches.
    """
    cleaned = raw.strip().strip('"').strip("'").strip()

    # Exact match
    for task in TaskType:
        if task.value == cleaned:
            return task

    # Substring match (model might emit extra text around the value)
    for task in TaskType:
        if task.value in cleaned:
            return task

    logger.warning(
        "Intent parser returned unrecognised output '%s'; "
        "falling back to SINGLE_VQA.",
        raw,
    )
    return TaskType.SINGLE_VQA
