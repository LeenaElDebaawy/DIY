import re

from app.schemas.analysis import Stage3Output

# Em / en dashes read as "AI-generated". We normalise them to plain punctuation
# in EVERY model output, regardless of source, so nothing dash-styled reaches the
# user. The static prompts and UI are also cleaned at source. These functions use
# unicode escapes so this code itself stays free of the literal characters.
_DASHES = (chr(0x2014), chr(0x2013), chr(0x2015))  # em, en, horizontal-bar dashes


def _strip_em_dashes(s: str) -> str:
    """Replace em/en dashes with plain punctuation: ' X ' -> ', ', bare X -> '-'."""
    if not isinstance(s, str) or not s:
        return s
    for d in _DASHES:
        s = s.replace(" " + d + " ", ", ")   # clause separator -> comma
        s = s.replace(d, "-")                 # any remaining -> hyphen
    return s


def _normalise_dashes_deep(obj):
    """Recursively apply _strip_em_dashes to every string in a dict / list / str."""
    if isinstance(obj, str):
        return _strip_em_dashes(obj)
    if isinstance(obj, list):
        return [_normalise_dashes_deep(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalise_dashes_deep(v) for k, v in obj.items()}
    return obj


# Tier tags live in Stage 2 output where the UI renders them as styled chips,
# but Llama (Stage 3) sometimes copies the literal "[Tier]" prefix into the
# step prose ("Use a [Likely] chisel ..."). The Stage 3 prompt forbids this,
# this regex is a deterministic backstop that strips the tags from step
# titles/descriptions/warnings and the global safety notes after Stage 3 has
# returned. It only affects PROSE strings, never the Stage 2 list items where
# the tags are meaningful structured data.
_TIER_TAG_RE = re.compile(r"\s*\[(?:Confirmed|Likely|Verify First|Do Not Buy Yet)\]\s*")


def _strip_tier_tags(s: str) -> str:
    """Remove '[Confirmed]' / '[Likely]' / '[Verify First]' / '[Do Not Buy Yet]'
    fragments from a prose string, collapsing the resulting whitespace and
    fixing the most common 'use a  chisel' / 'a  vacuum' double-space + dangling
    article artefacts."""
    if not isinstance(s, str) or not s:
        return s
    cleaned = _TIER_TAG_RE.sub(" ", s)
    # Tidy up double spaces and "a/an" left dangling before a now-bare noun
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # "use a  chisel" already collapsed above; nothing further needed.
    return cleaned


def _scrub_stage3_prose(s3: Stage3Output) -> int:
    """Strip tier tags from every prose string in a Stage3Output, in place.
    Returns the number of strings actually modified (for transparency / logging)."""
    changed = 0
    for step in (s3.steps or []):
        for attr in ("title", "description", "warning"):
            v = getattr(step, attr, None)
            if isinstance(v, str):
                cleaned = _strip_tier_tags(v)
                if cleaned != v:
                    setattr(step, attr, cleaned)
                    changed += 1
    new_notes = []
    for n in (s3.safety_notes or []):
        if isinstance(n, str):
            cleaned = _strip_tier_tags(n)
            if cleaned != n:
                changed += 1
            new_notes.append(cleaned)
        else:
            new_notes.append(n)
    s3.safety_notes = new_notes
    if isinstance(s3.when_to_call_professional, str):
        cleaned = _strip_tier_tags(s3.when_to_call_professional)
        if cleaned != s3.when_to_call_professional:
            s3.when_to_call_professional = cleaned
            changed += 1
    return changed


def _clean_json(text: str) -> str:
    """
    Strip markdown fences and extract the first JSON object from the response.
    Local models sometimes wrap JSON in ```json ... ``` or add prose before/after.
    """
    text = text.strip()

    # Remove ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find the outermost {...} block in case the model added prose around the JSON
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]

    return text


def _heal_truncated_json(text: str) -> str:
    """
    Attempt to repair JSON that was cut off mid-generation (truncated by num_predict
    or num_ctx limits).  Closes any unclosed strings, arrays, and objects.

    This is a best-effort recovery, it will not always produce semantically
    correct JSON, but it often lets the Pydantic validator extract partial
    data rather than failing completely.
    """
    text = text.rstrip()
    if not text:
        return text

    # If we're inside an unclosed string, close it
    # Heuristic: count unescaped double-quotes; if odd, we're mid-string
    in_string   = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        text += '"'     # close the open string

    # Close open arrays and objects in reverse order of opening
    stack = []
    in_string   = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            continue
        if ch == '"' and in_string:
            in_string = False
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    # Append the closing tokens in reverse
    text += "".join(reversed(stack))
    return text
