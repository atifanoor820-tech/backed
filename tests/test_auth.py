"""Sign-up, login and per-user data: users see only their own conversations, trading accounts and
ShowMyTrades profiles; the admin sees everyone's."""

import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(settings, monkeypatch):
    settings(admin_email="boss@example.com", admin_password="admin-pass-123")
    import main
    import services.agent_api.main as api
    from database.models import Base
    from database.session import engine

    Base.metadata.create_all(engine)  # the suite builds the schema directly, not with Alembic
    monkeypatch.setattr(api, "migrate", lambda: None)
    with TestClient(main.app) as c:
        yield c


def signup(c, email=None, password="secret-123"):
    email = email or f"u{uuid.uuid4().hex[:8]}@example.com"
    r = c.post("/api/auth/signup", json={"email": email, "password": password, "name": "Test"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}, r.json()["user"]


def test_everything_needs_login_except_health_and_auth(client):
    assert client.get("/api/health").status_code == 200
    for path in ("/api/status", "/api/conversations", "/api/tasks", "/api/dashi", "/api/memories"):
        assert client.get(path).status_code == 401, path


def test_signup_login_logout(client):
    email = f"a{uuid.uuid4().hex[:6]}@example.com"
    h, user = signup(client, email)
    assert user["role"] == "user" and user["email"] == email
    assert client.get("/api/auth/me", headers=h).json()["id"] == user["id"]

    assert client.post("/api/auth/signup", json={"email": email, "password": "another-123"}).status_code == 400
    assert client.post("/api/auth/login", json={"email": email, "password": "wrong-pass"}).status_code == 401
    r = client.post("/api/auth/login", json={"email": email.upper(), "password": "secret-123"})
    assert r.status_code == 200

    client.post("/api/auth/logout", headers=h)
    assert client.get("/api/auth/me", headers=h).status_code == 401


def test_users_see_only_their_own_accounts_and_profiles(client):
    h1, _ = signup(client)
    h2, _ = signup(client)

    fresh = client.get("/api/dashi", headers=h1).json()
    assert fresh["trading_accounts"] == [] and all(not p["email"] for p in fresh["profiles"])

    client.post("/api/dashi/accounts/import", headers=h1, json={"text": "11112222 / pw1 / FxPro-MT5 Demo"})
    profiles = fresh["profiles"]
    profiles[0] = {**profiles[0], "email": "one@smt.test", "password": "p1"}
    client.put("/api/dashi", headers=h1, json={"profiles": profiles})

    mine = client.get("/api/dashi", headers=h1).json()
    theirs = client.get("/api/dashi", headers=h2).json()
    assert [a["account_number"] for a in mine["trading_accounts"]] == ["11112222"]
    assert mine["profiles"][0]["email"] == "one@smt.test"
    assert theirs["trading_accounts"] == [] and theirs["profiles"][0]["email"] == ""


def test_conversations_are_private_but_admin_sees_all(client):
    h1, u1 = signup(client)
    h2, _ = signup(client)
    c1 = client.post("/api/conversations", headers=h1).json()["id"]

    assert c1 in [c["id"] for c in client.get("/api/conversations", headers=h1).json()]
    assert c1 not in [c["id"] for c in client.get("/api/conversations", headers=h2).json()]
    assert client.get(f"/api/conversations/{c1}", headers=h2).status_code == 404
    assert client.delete(f"/api/conversations/{c1}", headers=h2).status_code == 404

    tok = client.post("/api/auth/login", json={"email": "boss@example.com", "password": "admin-pass-123"}).json()
    ha = {"Authorization": f"Bearer {tok['token']}"}
    assert tok["user"]["role"] == "admin"
    everyone = client.get("/api/conversations", headers=ha).json()
    assert c1 in [c["id"] for c in everyone] and all("owner" in c for c in everyone)
    only_u1 = client.get(f"/api/conversations?as_user={u1['id']}", headers=ha).json()
    assert [c["id"] for c in only_u1] == [c1]

    client.post(f"/api/dashi/accounts/import?as_user={u1['id']}", headers=ha, json={"text": "33334444 / x / FxPro-MT5 Demo"})
    assert "33334444" in [a["account_number"] for a in client.get("/api/dashi", headers=h1).json()["trading_accounts"]]

    users = client.get("/api/admin/users", headers=ha).json()
    assert u1["id"] in [u["id"] for u in users]
    assert client.get("/api/admin/users", headers=h1).status_code == 403


def test_only_admin_changes_shared_settings(client):
    h, _ = signup(client)
    assert client.patch("/api/settings", headers=h, json={"autonomy": "full"}).status_code == 403
    assert client.get("/api/settings", headers=h).status_code == 200


def test_a_task_works_on_its_owners_accounts(client, settings):
    import time

    settings(dashi_only=True)
    h1, _ = signup(client)
    h2, _ = signup(client)
    client.post("/api/dashi/accounts/import", headers=h1, json={"text": "55556666 / a / FxPro-MT5 Demo\n77778888 / b / FxPro-MT5 Demo"})

    def ask(h):
        tid = client.post("/api/tasks", headers=h, json={"text": "ShowMyTrades accounts dikhao"}).json()["task_id"]
        for _ in range(100):
            t = client.get(f"/api/tasks/{tid}", headers=h).json()
            if t["status"] in ("completed", "failed", "blocked"):
                return t["result"] or ""
            time.sleep(0.1)
        return ""

    assert "Trading Accounts: 2" in ask(h1)
    assert "Trading Accounts: 0" in ask(h2)
