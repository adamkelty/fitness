"""One-time OAuth setup for Strava, plus the token refresh used on every pipeline run.

Strava's OAuth flow: the user approves access in a browser, Strava redirects to a
localhost URL with a `code` query param (the page fails to load - that's expected,
nothing is listening on localhost). That code is exchanged once for a refresh_token,
which is long-lived and gets stored in .env. Every pipeline run then exchanges the
refresh_token for a short-lived access_token (expires after 6 hours).
"""

import requests

from pipeline.config import STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN

AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"


def get_authorize_url() -> str:
    params = (
        f"client_id={STRAVA_CLIENT_ID}"
        "&redirect_uri=http://localhost/exchange_token"
        "&response_type=code"
        "&approval_prompt=force"
        "&scope=activity:read_all"
    )
    return f"{AUTHORIZE_URL}?{params}"


def exchange_code_for_tokens(code: str) -> dict:
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
    )
    response.raise_for_status()
    return response.json()


def get_access_token() -> str:
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "refresh_token": STRAVA_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _run_cli() -> None:
    print("1. Open this URL in your browser and approve access:\n")
    print(f"   {get_authorize_url()}\n")
    print("2. You'll land on a 'page not found' localhost URL - that's expected.")
    print("   Copy the `code` query param value from that URL's address bar.\n")
    code = input("Paste the code here: ").strip()

    tokens = exchange_code_for_tokens(code)
    print("\nSuccess! Add this to your .env file:\n")
    print(f"STRAVA_REFRESH_TOKEN={tokens['refresh_token']}")


if __name__ == "__main__":
    _run_cli()
