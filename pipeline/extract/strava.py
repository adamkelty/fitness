import requests

from pipeline.auth.strava_auth import get_access_token

API_BASE = "https://www.strava.com/api/v3"
PAGE_SIZE = 200
STREAM_KEYS = "time,distance,heartrate,velocity_smooth,altitude"


class RateLimitError(Exception):
    """Raised when Strava returns 429 (read rate limit exceeded)."""


def get_activities() -> list[dict]:
    headers = {"Authorization": f"Bearer {get_access_token()}"}
    activities = []
    page = 1

    while True:
        response = requests.get(
            f"{API_BASE}/athlete/activities",
            headers=headers,
            params={"per_page": PAGE_SIZE, "page": page},
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        activities.extend(batch)
        page += 1

    return activities


def get_gear(gear_id: str) -> dict:
    headers = {"Authorization": f"Bearer {get_access_token()}"}
    response = requests.get(f"{API_BASE}/gear/{gear_id}", headers=headers)
    response.raise_for_status()
    return response.json()


class StravaReader:
    """Fetches per-activity detail and streams, tracking Strava's read rate limit.

    Strava's binding limit is 100 reads / 15 min (and 1000 / day). After each
    call this reads the `x-readratelimit-usage` header so a caller can check
    `window_used` and stop before hitting the cap. A 429 raises RateLimitError.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {get_access_token()}"
        self.window_used = 0
        self.day_used = 0

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = self.session.get(f"{API_BASE}{path}", params=params)
        if response.status_code == 429:
            raise RateLimitError(path)
        response.raise_for_status()

        # The streams endpoint omits the usage header, so count every read
        # locally and prefer the server's authoritative number when present.
        self.window_used += 1
        self.day_used += 1
        usage = response.headers.get("x-readratelimit-usage")
        if usage:
            self.window_used, self.day_used = (int(x) for x in usage.split(","))
        return response.json()

    def get_activity_detail(self, activity_id: int) -> dict:
        return self._get(f"/activities/{activity_id}")

    def get_activity_streams(self, activity_id: int) -> dict:
        return self._get(
            f"/activities/{activity_id}/streams",
            params={"keys": STREAM_KEYS, "key_by_type": "true"},
        )
