"""Real YouTube transcript fetching -- for "learn X from this video/from
YouTube" requests, both one-off (main.py's youtube_learn_request axis) and
inside mastery job sessions (MASTERY_PROMPT_TEMPLATE tells the Hermes
agent it has this available for skills that are commonly YouTube-taught,
e.g. editing, trading, SMMA).

Verified against the REAL installed youtube-transcript-api==1.2.4 API
(instance-based .fetch()/.list()) -- NOT the old static get_transcript()
classmethod that's still what most blog posts/training data show. This
project has been burned by exactly that kind of unverified-API assumption
before, so: checked with the actual package installed, not guessed.

Known real limitation, not hidden: this library is unofficial and gets
IP-blocked on cloud hosts including exactly the kind of host Sandy runs
on (Hugging Face Spaces). It may simply fail in production even though it
works fine here/locally -- that needs a real check on the live Space, not
an assumption either way.
"""
import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

_ID_PATTERNS = [
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
]


def extract_video_id(text: str) -> str | None:
    """Pulls an 11-char YouTube video ID out of a URL found anywhere in
    free text (a full chat message, not just a bare URL). Regex-based on
    purpose -- don't trust an LLM to copy an ID character-for-character
    without typos."""
    for pattern in _ID_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    # bare 11-char ID with no URL wrapper -- only trust this if the message
    # actually mentions youtube/video, since an unguarded match risks
    # false-positiving on order IDs, hashes, or other random 11-char tokens
    if re.search(r"\byoutube\b|\bvideo\b", text, re.I):
        m = re.search(r"\b([A-Za-z0-9_-]{11})\b", text)
        if m:
            return m.group(1)
    return None


def transcript(video_id: str, languages: tuple[str, ...] = ("en", "hi")) -> str:
    """Real transcript text for a video, most-preferred language first.
    Raises a plain RuntimeError with an honest reason on failure (no
    captions, disabled, blocked, video gone) -- caller decides how to
    tell Ruk, this doesn't paper over it."""
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except NoTranscriptFound:
        raise RuntimeError(f"No transcript in {languages} for this video -- it may only have other languages, or none at all.")
    except TranscriptsDisabled:
        raise RuntimeError("Captions are disabled on this video -- can't read what wasn't captioned.")
    except VideoUnavailable:
        raise RuntimeError("Video isn't accessible (private, deleted, or region-locked).")
    except Exception as e:
        # Real, known risk (see module docstring): unofficial library,
        # cloud IPs get blocked. Don't disguise that as "no transcript".
        raise RuntimeError(f"Transcript fetch failed -- possibly IP-blocked from this server: {e!r}")
    return " ".join(s.text for s in fetched)
