#!/usr/bin/env python3
"""Re-authorize Spotify with updated scopes. Run on your Mac.

Reads client_id/client_secret from config.yaml (never hardcode them here —
this file is tracked in git)."""

import os
import urllib.parse
import webbrowser

import requests
import yaml

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path) as _f:
    _spotify = (yaml.safe_load(_f) or {}).get("spotify", {})

CLIENT_ID = _spotify.get("client_id", "")
CLIENT_SECRET = _spotify.get("client_secret", "")
REDIRECT_URI = "https://example.com/callback"

if not CLIENT_ID or not CLIENT_SECRET:
    raise SystemExit("ERROR: spotify.client_id / client_secret not set in config.yaml")

SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-read-recently-played",
])

def main():
    auth_url = (
        f"https://accounts.spotify.com/authorize"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={urllib.parse.quote(REDIRECT_URI)}"
        f"&scope={urllib.parse.quote(SCOPES)}"
    )

    print("Opening browser for Spotify authorization...\n")
    webbrowser.open(auth_url)

    print("After approving, the browser will redirect to a URL that won't load.")
    print("That's OK — copy the FULL URL from the address bar and paste it here.\n")
    url = input("Paste the redirect URL: ").strip()

    query = urllib.parse.urlparse(url).query
    params = urllib.parse.parse_qs(query)
    code = params.get("code", [None])[0]

    if not code:
        print("ERROR: No authorization code found in that URL.")
        return

    print("Exchanging for tokens...")
    resp = requests.post("https://accounts.spotify.com/api/token", data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} — {resp.text}")
        return

    tokens = resp.json()
    refresh_token = tokens.get("refresh_token")

    print(f"\n{'='*60}")
    print(f"NEW REFRESH TOKEN:")
    print(f"{refresh_token}")
    print(f"{'='*60}")
    print(f"\nUpdate this in config.yaml under spotify.refresh_token")

if __name__ == "__main__":
    main()
