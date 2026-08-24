"""Protects specific classes of content from being cut mid-string when
llm.fit_to_budget() has to truncate an oversized message.

Adapted from studying OmniRoute's preservation engine
(open-sse/services/compression/preservation.ts) -- NOT a port. Its real
version protects 15+ content classes (math blocks, LaTeX, Typst
directives, markdown tables, const-case identifiers, dotted method
calls...) because it sits in front of arbitrary, unknown request
bodies from many different client tools. Sandy generates its own
prompts and knows what's actually in them, so this keeps only the
classes actually relevant to Sandy's real content: fenced code blocks,
inline code, URLs, file paths, and error messages/stack traces. Same
placeholder-and-restore mechanism as the original (extract before
truncating, put back after) -- that mechanism, not the exhaustive
pattern list, is the part worth borrowing.

This is deliberately NOT prompt compression -- nothing here shrinks
content that fits. It only stops truncation (which already existed in
fit_to_budget as the last-resort safety valve for oversized content)
from slicing through the middle of a code block or a URL.
"""
import re

_SENTINEL_PREFIX = "\x00SANDY_PRESERVE"

# Order matters: fenced code blocks first (so a URL or error message
# *inside* a code fence gets protected as part of the fence, not
# separately -- avoids double-wrapping and placeholder corruption).
_PATTERNS = [
    ("fenced_code", re.compile(r"```.*?```", re.DOTALL)),
    ("inline_code", re.compile(r"`[^`\n]+`")),
    ("url", re.compile(r"https?://[^\s)\]\"'>]+")),
    ("file_path", re.compile(r"(?:^|\s)\.{0,2}/[A-Za-z0-9_@./-]+")),
    ("error_message", re.compile(
        r"\b(?:TypeError|ReferenceError|SyntaxError|ValueError|KeyError|"
        r"AttributeError|IndexError|Exception|Error):[^\n]+"
    )),
]


def extract_preserved(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replaces protected spans with opaque placeholders. Returns
    (text_with_placeholders, [(placeholder, original_content), ...]).
    Safe to call on any text -- if nothing matches, returns it unchanged
    with an empty block list."""
    blocks: list[tuple[str, str]] = []

    def _replace(kind: str):
        def _sub(m: re.Match) -> str:
            placeholder = f"{_SENTINEL_PREFIX}_{len(blocks)}\x00"
            blocks.append((placeholder, m.group(0)))
            return placeholder
        return _sub

    for kind, pattern in _PATTERNS:
        text = pattern.sub(_replace(kind), text)
    return text, blocks


def restore_preserved(text: str, blocks: list[tuple[str, str]]) -> str:
    """Inverse of extract_preserved. Order-independent -- each
    placeholder is unique, so restoring in any order is safe."""
    for placeholder, original in blocks:
        text = text.replace(placeholder, original)
    return text


def truncate_preserving(text: str, max_chars: int, note: str) -> str:
    """What fit_to_budget's oversized-content branch now calls instead
    of a bare text[:room] slice. Extracts protected spans first, does
    the truncation against the placeholder'd text (placeholders are
    short and fixed-length, so this doesn't blow the budget), then
    restores whatever protected blocks survived intact within the kept
    portion. A placeholder that got cut off mid-truncation is dropped
    (its content is gone either way -- the goal is never corrupting a
    KEPT block, not guaranteeing every block survives)."""
    if len(text) <= max_chars:
        return text
    placeholdered, blocks = extract_preserved(text)
    cut = placeholdered[:max_chars]
    # Each COMPLETE placeholder contains exactly 2 null bytes (open +
    # close). An ODD null-byte count in `cut` means the last one was
    # opened but never closed -- the slice landed mid-placeholder.
    # Drop that dangling fragment entirely rather than leak a raw
    # sentinel byte into the prompt. (An earlier version tried to be
    # clever about where exactly to cut and got it wrong -- verified
    # by a direct test that caught a leaked \x00 byte before this fix.)
    if cut.count("\x00") % 2 == 1:
        cut = cut[:cut.rfind("\x00")]
    restored = restore_preserved(cut, blocks)
    return restored + note
