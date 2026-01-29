"""Utility for creating requests sessions with retry logic."""

from datetime import timedelta
from http import HTTPStatus

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_retry_session(
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    backoff_max: timedelta = timedelta(seconds=60),
) -> requests.Session:
    """Create a requests session with retry logic.

    Args:
        total_retries: maximum number of retries
        backoff_factor: factor for exponential backoff
        backoff_max: maximum backoff time

    Returns:
        configured requests session
    """
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        backoff_max=backoff_max.total_seconds(),
        allowed_methods=["GET", "POST"],
        status_forcelist=[
            HTTPStatus.TOO_MANY_REQUESTS,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        ],
        raise_on_status=False,
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
