"""Logging setup. Never log image bytes, API keys, or raw request payloads."""

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=_LOG_FORMAT, force=True)
    # The OpenAI/httpx clients log request details at DEBUG/INFO; keep them quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
