import requests

from pipeline.auth.strava_auth import get_access_token

API_BASE = "https://www.strava.com/api/v3"
PAGE_SIZE = 200


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
