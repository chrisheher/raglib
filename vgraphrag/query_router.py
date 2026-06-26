"""
query_router.py
===============
Stage 3 (prefix): Query classification before operator dispatch.

Classifies each incoming query as:
  SPECIFIC  — character, plot, named work, intertextual fact
  ABSTRACT  — thematic, cross-author, systemic, philosophical

Uses a Claude API call with few-shot prompt and JSON output.
Falls back to keyword heuristics if the API call fails.
"""

import json
import re
from typing import Literal, Optional

import anthropic
from dotenv import load_dotenv
import os

from .prompts import ROUTER_SYSTEM, ROUTER_USER

load_dotenv()

MODEL = "claude-sonnet-4-6"
QueryMode = Literal["specific", "abstract"]


# ─────────────────────────────────────────────────────────────
# KEYWORD FALLBACK
# Fast heuristic used when API is unavailable.
# ─────────────────────────────────────────────────────────────

_SPECIFIC_SIGNALS = [
    "who is", "who are", "who does", "who foils",
    "how does", "what does", "what is the relationship between",
    "compare", "contrast", "between", "versus", "vs",
    "character", "narrator", "protagonist", "antagonist",
    "quote", "passage", "scene", "chapter",
    "specifically", "in the novel", "in the poem", "in the play",
    "alludes to", "references", "intertextual", "influence",
    "example of", "instance of",
]

_ABSTRACT_SIGNALS = [
    "throughout", "across", "generally", "broadly", "in general",
    "theme", "motif", "pattern", "recurring", "common",
    "what role does", "what drives", "what connects",
    "tradition", "genre", "movement", "school of",
    "relationship between creativity", "relationship between consciousness",
    "how is", "why does", "what is the significance of",
    "overall", "corpus", "these texts", "multiple works",
    "philosophy of", "theory of",
]


def _keyword_classify(query: str) -> QueryMode:
    q = query.lower()
    spec_score = sum(1 for sig in _SPECIFIC_SIGNALS if sig in q)
    abst_score = sum(1 for sig in _ABSTRACT_SIGNALS if sig in q)
    return "abstract" if abst_score > spec_score else "specific"


# ─────────────────────────────────────────────────────────────
# LLM ROUTER
# ─────────────────────────────────────────────────────────────

def classify_query(
    query: str,
    client: Optional[anthropic.Anthropic] = None,
    use_llm: bool = True,
) -> QueryMode:
    """
    Classify a query as 'specific' or 'abstract'.

    Args:
        query:    The user question.
        client:   An Anthropic client instance (created if None).
        use_llm:  If False, uses keyword heuristic only (faster, cheaper).

    Returns:
        'specific' or 'abstract'
    """
    if not use_llm:
        return _keyword_classify(query)

    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    user_prompt = ROUTER_USER.format(query=query)

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=64,
            system=ROUTER_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

        parsed = json.loads(raw)
        classification = parsed.get("classification", "").upper()

        if classification in ("SPECIFIC", "ABSTRACT"):
            return classification.lower()  # type: ignore[return-value]

        # Unexpected value — fall back to keyword
        return _keyword_classify(query)

    except Exception as e:
        # Any failure → fast fallback
        print(f"  Router LLM failed ({e}), using keyword heuristic")
        return _keyword_classify(query)
