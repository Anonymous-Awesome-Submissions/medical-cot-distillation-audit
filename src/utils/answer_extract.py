"""Regex MCQ answer-letter extractor for medical CoT outputs (A-D)."""
import re


def extract_answer(text: str) -> str:
    """Extract answer letter from generated text.

    Uses a cascade of regex patterns from most specific to most general.
    Designed to achieve >95% extraction rate on medical MCQ CoT outputs.
    """
    # Tier 1: Explicit answer statements (highest confidence)
    tier1 = [
        r'[Tt]he\s+answer\s+is\s*\(?([A-D])\)?',
        r'[Cc]orrect\s+answer\s*(?:is|:)\s*\(?([A-D])\)?',
        r'[Aa]nswer\s*:\s*\(?([A-D])\)?',
        r'\b([A-D])\)\s*is\s+(?:the\s+)?correct',
    ]
    for pattern in tier1:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # Tier 2: Conclusion phrases (medium confidence)
    tier2 = [
        r'(?:[Tt]herefore|[Tt]hus|[Hh]ence|[Ss]o),?\s+(?:the\s+)?(?:answer|correct\s+option)\s+(?:is|would\s+be)\s+\(?([A-D])\)?',
        r'(?:choose|select|pick|go\s+with)\s+\(?([A-D])\)?',
        r'(?:I\s+would\s+(?:choose|select|answer))\s+\(?([A-D])\)?',
        r'[Oo]ption\s+([A-D])\s+is\s+(?:the\s+)?(?:correct|best|most)',
        r'\*\*([A-D])\*\*',  # Bold markdown answer
    ]
    for pattern in tier2:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    # Tier 3: Last occurrence in final portion of text (lower confidence)
    tail = text[-200:] if len(text) > 200 else text
    match = re.search(r'\(([A-D])\)', tail)
    if match:
        last = match
        for m in re.finditer(r'\(([A-D])\)', tail):
            last = m
        return last.group(1).upper()

    # Standalone letter at the very end
    match = re.search(r'\b([A-D])\b\s*[.\s]*$', text.strip())
    if match:
        return match.group(1).upper()

    return ""
