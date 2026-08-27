"""Shared VLM output parsing logic for all backends."""

from __future__ import annotations

import json
import re

from video_eval.core.exceptions import VLMOutputParseError
from video_eval.core.schemas import EvidenceItem, VLMResult


def parse_vlm_output(raw_output: str) -> VLMResult:
    """Extract structured evaluation from raw VLM text output.

    Searches for a JSON object containing at minimum a 'level' field (1-5).
    Maps level to a normalized score via: score = (level - 1) / 4.

    Args:
        raw_output: Raw text output from a VLM backend.

    Returns:
        Parsed VLMResult with normalized score and metadata.

    Raises:
        VLMOutputParseError: If no valid JSON object with 'level' can be extracted.
    """
    # Try to find a JSON object in the output
    json_obj = _extract_json_object(raw_output)
    if json_obj is None:
        raise VLMOutputParseError(
            f"No valid JSON object found in VLM output: {raw_output[:200]}",
            raw_output=raw_output,
        )

    # Extract and validate level
    level = json_obj.get("level")
    if level is None:
        raise VLMOutputParseError(
            f"JSON object missing 'level' field: {raw_output[:200]}",
            raw_output=raw_output,
        )

    try:
        level = int(level)
    except (TypeError, ValueError):
        raise VLMOutputParseError(
            f"'level' is not a valid integer: {level!r}",
            raw_output=raw_output,
        )

    if level < 1 or level > 5:
        raise VLMOutputParseError(
            f"'level' must be between 1 and 5, got {level}",
            raw_output=raw_output,
        )

    # Map level (1-5) to score (0.0-1.0)
    score = (level - 1) / 4.0

    # Extract optional fields
    reasoning = json_obj.get("reasoning", "")
    suggestion = json_obj.get("suggestion", "")

    # Parse evidence items
    evidence_raw = json_obj.get("evidence", [])
    evidence: list[EvidenceItem] = []
    if isinstance(evidence_raw, list):
        for item in evidence_raw:
            if isinstance(item, dict):
                evidence.append(
                    EvidenceItem(
                        modality=item.get("modality", "visual"),
                        timestamp=item.get("timestamp"),
                        detail=item.get("detail", ""),
                    )
                )

    return VLMResult(
        score=score,
        reasoning=reasoning,
        evidence=evidence,
        suggestion=suggestion,
        raw_output=raw_output,
    )


def _extract_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from text.

    Tries multiple strategies:
    1. Direct JSON parse of the entire text.
    2. Find content between first '{' and last '}' and parse.
    3. Regex-based extraction of JSON-like blocks.

    Returns:
        Parsed dict or None if extraction fails.
    """
    # Strategy 1: try parsing entire text as JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: find the outermost { ... } block
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: regex for JSON-like blocks (handles escaped characters)
    pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
    for match in pattern.finditer(text):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue

    return None
