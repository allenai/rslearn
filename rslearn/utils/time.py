from datetime import datetime, timedelta
from typing import Generator


def daterange(
    start_time: datetime, end_time: datetime
) -> Generator[datetime, None, None]:
    for n in range(int((end_time - start_time).days)):
        yield start_time + timedelta(n)
