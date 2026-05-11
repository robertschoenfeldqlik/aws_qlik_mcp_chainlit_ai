"""OAuth PKCE flow for Qlik Cloud MCP (streamable-http transport).

Matches the working LibreChat configuration:
- Transport: streamable-http
- Header: X-Agent-Id
- OAuth: Authorization Code + PKCE (S256)
- Scopes: user_default mcp:execute
- No client secret (public/native client)
"""

import base64
import hashlib
import html as html_mod
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from loguru import logger


# ---------------------------------------------------------------------------
# State Stores (module-level — shared between FastAPI routes and Chainlit)
# ---------------------------------------------------------------------------

@dataclass
class PendingOAuth:
    tenant_url: str
    client_id: str
    code_verifier: str
    redirect_uri: str
    session_id: str
    created_at: float = field(default_factory=time.time)


# In-flight OAuth flows, keyed by `state` (random per flow).
pending_flows: dict[str, PendingOAuth] = {}

# Completed token exchanges awaiting JS pickup, keyed by `state`.
completed_tokens: dict[str, dict] = {}

# Tokens delivered by JS, awaiting Chainlit's on_message. Keyed by the
# Chainlit websocket session ID that the JS reported in /auth/qlik/start.
pending_connections: dict[str, dict] = {}

# How long a state entry can sit before being garbage-collected (seconds).
_STATE_TTL = 600
_PENDING_TTL = 1800


def _cleanup() -> None:
    """Drop stale entries from all three state stores."""
    now = time.time()
    for state, flow in list(pending_flows.items()):
        if now - flow.created_at > _STATE_TTL:
            del pending_flows[state]
    for state, token in list(completed_tokens.items()):
        if now - token.get("t", 0) > _STATE_TTL:
            del completed_tokens[state]
    for sid, conn in list(pending_connections.items()):
        if now - conn.get("t", 0) > _PENDING_TTL:
            del pending_connections[sid]


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------

def _verifier() -> str:
    return secrets.token_urlsafe(48)


def _challenge(v: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()


# ---------------------------------------------------------------------------
# Tenant URL validation (mirrors app._validate_qlik_tenant)
# ---------------------------------------------------------------------------

def _validate_tenant(tenant_url: str) -> str:
    parsed = urlparse(tenant_url)
    if parsed.scheme != "https":
        raise ValueError("Tenant URL must use https://")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Tenant URL missing hostname")
    extra = os.getenv("QLIK_TENANT_ALLOWLIST", "")
    allowed = [".qlikcloud.com", ".qlik-stage.com"] + [
        s.strip().lower() for s in extra.split(",") if s.strip()
    ]
    if not any(host == s.lstrip(".") or host.endswith(s) for s in allowed):
        raise ValueError(f"Tenant host {host!r} not in QLIK_TENANT_ALLOWLIST")
    return f"https://{host}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def register_oauth_routes(app):
    router = APIRouter()

    @router.get("/auth/qlik/defaults")
    async def defaults(request: Request):
        return JSONResponse({
            "tenant_url": os.getenv("QLIK_TENANT_URL", ""),
            "client_id": os.getenv("QLIK_OAUTH_CLIENT_ID", ""),
        })

    @router.get("/auth/qlik/status")
    async def status(request: Request):
        _cleanup()
        state = request.query_params.get("state", "")
        if state in completed_tokens:
            token_data = completed_tokens.pop(state)
            return JSONResponse({
                "complete": True,
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token", ""),
                "tenant_url": token_data["tenant_url"],
                "client_id": token_data["client_id"],
                "session_id": token_data.get("session_id", ""),
            })
        return JSONResponse({"complete": False})

    @router.post("/auth/qlik/connect")
    async def connect(request: Request):
        """JS calls this after OAuth — stores token for app.py to pick up."""
        try:
            body = await request.json()
            key = (body.get("session_id") or "").strip()
            if not key:
                return JSONResponse({"error": "session_id required"}, 400)
            try:
                _validate_tenant(body["tenant_url"])
            except (KeyError, ValueError) as e:
                return JSONResponse({"error": str(e)}, 400)
            if not body.get("access_token"):
                return JSONResponse({"error": "access_token required"}, 400)
            pending_connections[key] = {
                "access_token": body["access_token"],
                "tenant_url": body["tenant_url"],
                "client_id": body.get("client_id", ""),
                "t": time.time(),
            }
            logger.info(f"Stored pending MCP connection for session {key[:8]}…")
            return JSONResponse({"ok": True})
        except Exception as e:
            logger.error(f"/auth/qlik/connect failed: {e}")
            return JSONResponse({"error": "internal error"}, 500)

    @router.get("/auth/qlik/start")
    async def start(request: Request):
        tenant_url = request.query_params.get("tenant_url", "")
        client_id = request.query_params.get("client_id", "")
        state = request.query_params.get("state", "")
        session_id = request.query_params.get("session_id", "")

        if not tenant_url or not client_id or not state:
            return HTMLResponse("<h2>Missing parameters</h2>", 400)

        try:
            tenant_url = _validate_tenant(tenant_url)
        except ValueError as e:
            return HTMLResponse(f"<h2>Invalid tenant URL</h2><p>{html_mod.escape(str(e))}</p>", 400)

        _cleanup()
        verifier = _verifier()
        challenge = _challenge(verifier)
        base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")
        redirect_uri = f"{base_url}/auth/qlik/callback"

        pending_flows[state] = PendingOAuth(
            tenant_url=tenant_url, client_id=client_id,
            code_verifier=verifier, redirect_uri=redirect_uri,
            session_id=session_id,
        )

        url = f"{tenant_url}/oauth/authorize?" + urllib.parse.urlencode({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "user_default mcp:execute",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
        logger.info(f"OAuth redirect to {tenant_url}")
        return RedirectResponse(url)

    @router.get("/auth/qlik/callback")
    async def callback(request: Request):
        error = request.query_params.get("error")
        if error:
            return HTMLResponse(_page(
                "Authentication Failed",
                html_mod.escape(request.query_params.get("error_description", error)),
                False,
            ))

        code = request.query_params.get("code", "")
        state = request.query_params.get("state", "")
        pending = pending_flows.pop(state, None)

        if not pending:
            return HTMLResponse(_page(
                "Session Expired",
                "Please try connecting again from the chat.", False,
            ), 400)

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    f"{pending.tenant_url}/oauth/token",
                    data={
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": pending.redirect_uri,
                        "client_id": pending.client_id,
                        "code_verifier": pending.code_verifier,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Token exchange failed: {e}")
                return HTMLResponse(_page(
                    "Token Exchange Failed",
                    html_mod.escape(str(e)), False,
                ))

        completed_tokens[state] = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "tenant_url": pending.tenant_url,
            "client_id": pending.client_id,
            "session_id": pending.session_id,
            "t": time.time(),
        }

        logger.info("OAuth completed, token stored for polling")
        return HTMLResponse(_page(
            "Connected to Qlik Cloud",
            "You can close this tab and return to the chat.", True,
        ))

    # Insert routes before Chainlit's catch-all
    for route in reversed(router.routes):
        app.routes.insert(0, route)
    logger.info(f"Registered {len(router.routes)} OAuth routes")


def _page(title, msg, ok):
    c = "#009845" if ok else "#d32f2f"
    i = "&#10003;" if ok else "&#10007;"
    return f"""<!DOCTYPE html><html><head><title>{title}</title>
<style>body{{font-family:'Source Sans 3',sans-serif;background:#0f1a24;color:#e0e0e0;display:flex;
justify-content:center;align-items:center;min-height:100vh;margin:0}}
.c{{text-align:center;padding:40px}}.i{{font-size:64px;color:{c}}}
h1{{color:{c}}}p{{color:#a0a0a0}}
.b{{margin-top:20px;padding:10px 24px;background:{c};color:white;border:none;border-radius:6px;cursor:pointer}}
</style></head>
<body><div class="c"><div class="i">{i}</div><h1>{title}</h1><p>{msg}</p>
<button class="b" onclick="window.close()">Close this tab</button></div>
<script>{"setTimeout(()=>window.close(),3000)" if ok else ""}</script></body></html>"""
