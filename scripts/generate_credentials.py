"""One-time setup: writes credentials.yaml for the login gate (app/auth.py).

Creates the demo accounts with random passwords, stores only their bcrypt
hashes plus a random cookie-signing key, and prints the plaintext passwords
once — note them down; they can't be recovered from the file.

Usage (from the project root, with the venv active):
    python scripts/generate_credentials.py            # refuses to overwrite
    python scripts/generate_credentials.py --force    # regenerate (old passwords stop working)
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.auth import CREDENTIALS_PATH, USER_ROLES  # noqa: E402

COOKIE_NAME = "squat_coach_auth"
COOKIE_EXPIRY_DAYS = 7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing credentials.yaml")
    args = parser.parse_args()

    if CREDENTIALS_PATH.exists() and not args.force:
        print(f"{CREDENTIALS_PATH} already exists — not overwriting. Use --force to regenerate.")
        return 1

    import streamlit_authenticator as stauth

    passwords = {username: secrets.token_urlsafe(9) for username in USER_ROLES}
    usernames = {
        username: {
            "email": f"{username}@example.com",
            "first_name": username.split("_")[0].capitalize(),
            "last_name": "Demo",
            "password": stauth.Hasher.hash(pw),
        }
        for username, pw in passwords.items()
    }
    config = {
        "credentials": {"usernames": usernames},
        "cookie": {"name": COOKIE_NAME, "key": secrets.token_hex(32), "expiry_days": COOKIE_EXPIRY_DAYS},
    }

    CREDENTIALS_PATH.write_text(yaml.safe_dump(config, sort_keys=False))
    CREDENTIALS_PATH.chmod(0o600)

    print(f"Wrote {CREDENTIALS_PATH} (hashed passwords only).")
    print("Demo logins — shown once, note them down:")
    for username, pw in passwords.items():
        print(f"  {username:14} {pw}   ({USER_ROLES[username]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
