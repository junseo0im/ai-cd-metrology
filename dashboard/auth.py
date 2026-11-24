"""Single configured ADMIN; opaque grants live on this server, bound to a Streamlit session."""
from collections import deque
from dataclasses import dataclass
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
import tomllib

TOKEN_KEY = "_ida_admin_grant"
ITERATIONS = 600_000
SESSION_SECONDS = 8 * 60 * 60
FAILURE_LIMIT = 5
FAILURE_WINDOW = 60
COOLDOWN_SECONDS = 30


class AuthenticationError(ValueError):
    """Only non-sensitive, user-facing messages."""


def _hash_parts(encoded):
    try:
        algorithm, rounds, salt, digest = encoded.split("$")
        rounds = int(rounds)
        salt, digest = bytes.fromhex(salt), bytes.fromhex(digest)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= rounds <= 2_000_000:
            return None
        if not 16 <= len(salt) <= 64 or len(digest) != 32:
            return None
        return rounds, salt, digest
    except (ValueError, TypeError, AttributeError):
        return None


def hash_password(password):
    """For initial server setup only; never returns/stores the cleartext password."""
    if not isinstance(password, str) or not 12 <= len(password) <= 1024:
        raise ValueError("비밀번호는 12~1024자로 입력하세요.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return "pbkdf2_sha256$" + str(ITERATIONS) + "$" + salt.hex() + "$" + digest.hex()


def verify_password(password, encoded):
    parts = _hash_parts(encoded)
    if parts is None or not isinstance(password, str) or len(password) > 1024:
        return False
    rounds, salt, expected = parts
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(actual, expected)


@dataclass(frozen=True, repr=False)
class AdminConfig:
    username: str
    password_hash: str
    display_name: str = "ADMIN"

    @property
    def fingerprint(self):
        payload = json.dumps([self.username, self.password_hash, self.display_name])
        return hashlib.sha256(payload.encode("utf-8")).digest()


def configured_admin():
    """Read one complete server config; partial/malformed settings fail closed."""
    username = os.environ.get("IDA_ADMIN_USERNAME")
    password_hash = os.environ.get("IDA_ADMIN_PASSWORD_HASH")
    display_name = os.environ.get("IDA_ADMIN_DISPLAY_NAME", "ADMIN")
    if username is None and password_hash is None:
        import streamlit as st
        try:
            section = st.secrets.get("ida_admin", {})
            username = section.get("username")
            password_hash = section.get("password_hash")
            display_name = section.get("display_name", "ADMIN")
        except (OSError, ValueError, TypeError, AttributeError):
            return None
    if (not isinstance(username, str) or not username.strip() or len(username) > 128
            or not isinstance(display_name, str) or not display_name.strip()
            or len(display_name) > 80 or not _hash_parts(password_hash)):
        return None
    return AdminConfig(username.strip(), password_hash, display_name.strip())


class AdminSessions:
    """In-memory server grants and an account-wide failed-login throttle."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._lock = threading.RLock()
        self._grants = {}
        self._failures = deque()
        self._blocked_until = 0.0

    def _purge(self, now):
        self._grants = {key: value for key, value in self._grants.items() if value[2] > now}
        while self._failures and self._failures[0] <= now - FAILURE_WINDOW:
            self._failures.popleft()

    def login(self, config, session_id, username, password):
        if config is None or not session_id:
            raise AuthenticationError("관리자 계정 미설정 — GUEST 조회 모드입니다.")
        with self._lock:
            now = self.clock()
            self._purge(now)
            if now < self._blocked_until:
                raise AuthenticationError("로그인 시도가 일시 제한되었습니다. 잠시 후 다시 시도하세요.")
            password_ok = verify_password(password, config.password_hash)
            username_ok = isinstance(username, str) and hmac.compare_digest(
                username.encode("utf-8"), config.username.encode("utf-8"))
            if not (username_ok and password_ok):
                self._failures.append(now)
                if len(self._failures) >= FAILURE_LIMIT:
                    self._blocked_until = now + COOLDOWN_SECONDS
                raise AuthenticationError("로그인 정보를 확인하세요.")
            token = secrets.token_urlsafe(32)
            self._grants[hashlib.sha256(token.encode()).digest()] = (
                session_id, config.fingerprint, now + SESSION_SECONDS)
            self._failures.clear()
            return token

    def authorized(self, config, session_id, token):
        if config is None or not session_id or not isinstance(token, str) or len(token) > 256:
            return False
        with self._lock:
            self._purge(self.clock())
            grant = self._grants.get(hashlib.sha256(token.encode()).digest())
            return bool(grant and grant[0] == session_id
                        and hmac.compare_digest(grant[1], config.fingerprint))

    def revoke(self, session_id, token):
        if not isinstance(token, str) or len(token) > 256:
            return
        with self._lock:
            key = hashlib.sha256(token.encode()).digest()
            grant = self._grants.get(key)
            if grant and grant[0] == session_id:
                self._grants.pop(key, None)


_sessions = AdminSessions()


def _session_id():
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    context = get_script_run_ctx(suppress_warning=True)
    return context.session_id if context is not None else None


def is_admin():
    import streamlit as st
    session_id = _session_id()
    return bool(session_id and _sessions.authorized(
        configured_admin(), session_id, st.session_state.get(TOKEN_KEY)))


def current_user():
    config = configured_admin()
    if config is not None and is_admin():
        return {"role": "ADMIN", "display_name": config.display_name}
    return {"role": "GUEST", "display_name": "GUEST"}


def clear_record_drafts():
    import streamlit as st
    for key in list(st.session_state):
        if key.startswith("record_") and (key.endswith("_draft") or "_editor_" in key):
            st.session_state.pop(key, None)


def login(username, password):
    import streamlit as st
    session_id = _session_id()
    token = _sessions.login(configured_admin(), session_id, username, password)
    _sessions.revoke(session_id, st.session_state.get(TOKEN_KEY))
    st.session_state[TOKEN_KEY] = token
    clear_record_drafts()


def logout():
    import streamlit as st
    _sessions.revoke(_session_id(), st.session_state.pop(TOKEN_KEY, None))
    clear_record_drafts()


def setup_admin(path=None):
    """Interactive local setup: never overwrite an existing ADMIN or print credentials."""
    path = Path(path) if path is not None else Path(__file__).parent / ".streamlit" / "secrets.toml"
    if path.exists():
        with path.open("rb") as handle:
            existing = tomllib.load(handle)
        if "ida_admin" in existing:
            raise ValueError("관리자 설정이 이미 있습니다. 서버 관리자만 해당 설정을 변경하세요.")
    username = input("관리자 사용자명: ").strip()
    display = input("표시 이름 (빈 값: ADMIN): ").strip() or "ADMIN"
    if not username or len(username) > 128 or len(display) > 80:
        raise ValueError("사용자명 또는 표시 이름을 확인하세요.")
    password = getpass.getpass("관리자 비밀번호 (12자 이상): ")
    confirmation = getpass.getpass("비밀번호 확인: ")
    if not hmac.compare_digest(password.encode("utf-8"), confirmation.encode("utf-8")):
        raise ValueError("비밀번호 확인이 일치하지 않습니다.")
    encoded = hash_password(password)
    del password, confirmation
    section = "\n[ida_admin]\n" + "\n".join(
        key + " = " + json.dumps(value, ensure_ascii=False)
        for key, value in {"username": username, "password_hash": encoded, "display_name": display}.items()
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if path.exists() else "x", encoding="utf-8") as handle:
        handle.write(section)
    print("관리자 해시 설정을 저장했습니다. 비밀 파일을 Git에 추가하지 마세요.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IDA 단일 ADMIN 최초 설정")
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()
    if args.setup:
        try:
            setup_admin()
        except (OSError, ValueError):
            parser.exit(1, "설정을 저장하지 못했습니다. 기존 설정·입력값·파일 권한을 확인하세요.\n")
    else:
        parser.print_help()
