"""Login gate + role resolution.

Additive: does not import from or modify video.py, pipeline.py, pose.py,
bar_tracker.py, rep_detector.py, or assessor.py. Wraps the app's entry
point only — this is why adding it should not change any CV/pipeline
output for a given video.

Uses streamlit-authenticator (local, config-file based — no external
identity provider needed). Requires credentials.yaml, generated once by
scripts/generate_credentials.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import yaml
from yaml.loader import SafeLoader

CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / "credentials.yaml"

# Maps authenticated username -> role. Kept separate from
# streamlit-authenticator's own config so the role logic stays simple and
# inspectable in one place, rather than buried in the auth library's config shape.
USER_ROLES = {
    "athlete_demo": "athlete",
    "trainer_demo": "trainer",
}
DEFAULT_ROLE = "athlete"  # unmapped/unknown users get the more restricted role


def load_authenticator():
    import streamlit_authenticator as stauth  # deferred import — only needed if auth is actually used

    # Hosted deploys have no credentials.yaml on disk — recreate it from the
    # secret. `in st.secrets` raises (a FileNotFoundError subclass) when no
    # secrets.toml exists at all, which is the normal local setup.
    try:
        if not CREDENTIALS_PATH.exists() and "auth_credentials_yaml" in st.secrets:
            CREDENTIALS_PATH.write_text(st.secrets["auth_credentials_yaml"])
    except FileNotFoundError:
        pass

    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"{CREDENTIALS_PATH} not found. Run `python scripts/generate_credentials.py` once first."
        )
    with open(CREDENTIALS_PATH) as f:
        config = yaml.load(f, Loader=SafeLoader)

    return stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )


def require_login() -> Optional[str]:
    """Renders the login form if not authenticated and halts the script
    (st.stop()) until the user logs in. Call this FIRST in ui.py, before
    any existing session-state/pipeline logic runs.

    Returns the resolved role ("athlete" or "trainer") once authenticated.
    """
    authenticator = load_authenticator()
    authenticator.login(location="main")

    auth_status = st.session_state.get("authentication_status")

    if auth_status is False:
        st.error("Username/password is incorrect.")
        st.stop()
    elif auth_status is None:
        st.info("Demo accounts: athlete_demo / trainer_demo — see credentials.yaml for passwords.")
        st.stop()

    username = st.session_state.get("username")
    role = USER_ROLES.get(username, DEFAULT_ROLE)
    st.session_state.user_role = role

    with st.sidebar:
        st.caption(f"Logged in as **{username}** ({role})")
        authenticator.logout(location="sidebar")

    return role
