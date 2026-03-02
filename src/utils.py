"""
Security helpers - input sanitization, URL validation, and data masking.

I kept these as pure functions (no side effects, no state) so they're easy
to unit test and reuse across the API and engine modules.

The SSRF validation was something I added after reading about the
169.254.169.254 metadata endpoint attack vector on AWS - seemed like a
real risk since we accept arbitrary URLs from users for JD fetching.

Author: Mohammad Inayat Hussain
"""

import re
import html
import ipaddress
from urllib.parse import urlparse

import bleach


MAX_INPUT_LENGTH = 10_000

# We strip ALL html tags - no whitelist needed
ALLOWED_TAGS: list[str] = []
ALLOWED_ATTRS: dict[str, list[str]] = {}


def sanitize_input(text: str) -> str:
    """
    Clean user input before processing.

    This is the first thing that runs on any query, whether it comes from
    the API or the frontend. The pipeline is:
      1. Strip null bytes (can break string operations)
      2. Remove <script> blocks entirely (bleach strips tags but not content)
      3. Kill inline event handlers (onclick=, onerror=, etc.)
      4. Bleach strips remaining HTML tags
      5. Unescape HTML entities so &amp; becomes & again
      6. Truncate to 10k chars (anything longer is probably abuse)
      7. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")

    # Get rid of script blocks before bleach (bleach only strips the tags,
    # not the content between them)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Nuke event handler attributes
    text = re.sub(r"\bon\w+\s*=\s*[\"'][^\"']*[\"']", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bon\w+\s*=\s*\S+", " ", text, flags=re.IGNORECASE)

    text = bleach.clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    text = html.unescape(text)
    text = text[:MAX_INPUT_LENGTH]
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_url(url: str) -> bool:
    """
    Check if a URL is safe to fetch (SSRF prevention).

    I'm blocking:
      - Non-HTTP schemes (file://, ftp://, etc.)
      - Private/reserved IP ranges (127.x, 10.x, 172.16-31.x, 192.168.x)
      - Cloud metadata endpoints (169.254.169.254)
      - Localhost and .internal/.local hostnames
    """
    if not isinstance(url, str) or not url.strip():
        return False

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https"):
            return False

        host = parsed.hostname
        if not host:
            return False

        # Known dangerous hostnames
        blocked = {"localhost", "0.0.0.0", "metadata.google.internal", "metadata.google.com"}
        if host.lower() in blocked:
            return False

        # If it's a raw IP address, check if it's in a private range
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False
            if ip.is_link_local or ip.is_multicast:
                return False
            # AWS/GCP metadata endpoint specifically
            if str(ip) == "169.254.169.254":
                return False
        except ValueError:
            # Not an IP, it's a hostname - just check for suspicious suffixes
            if host.endswith(".internal") or host.endswith(".local"):
                return False

        return True

    except Exception:
        return False


def mask_sensitive(text: str) -> str:
    """
    Mask API keys / tokens for safe logging.

    Turns "AIzaSyD1234567890abcdefghijklmnop" into "AIza****mnop".
    Short strings and non-alphanumeric text pass through unchanged.
    """
    if not isinstance(text, str):
        return "***"

    text = text.strip()

    # Only mask if it looks like an API key (long, mostly alphanumeric)
    if len(text) < 12:
        return text

    alnum_ratio = sum(1 for c in text if c.isalnum() or c in "-_") / len(text)
    if alnum_ratio < 0.7:
        return text

    return f"{text[:4]}{'*' * (len(text) - 8)}{text[-4:]}"


def is_url(text: str) -> bool:
    """Quick check: does the text look like a URL?"""
    if not isinstance(text, str):
        return False
    return bool(re.match(r"https?://", text.strip(), re.IGNORECASE))
