"""Time-related utility functions."""

from collections.abc import Generator
from datetime import datetime, timedelta


def daterange(
    start_time: datetime, end_time: datetime
) -> Generator[datetime, None, None]:
    """Generator that yields each day between start_time and end_time."""
    for n in range(int((end_time - start_time).days)):
        yield start_time + timedelta(n)
