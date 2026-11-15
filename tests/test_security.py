"""Security and session tests (offline, no API key or network required)."""
import pytest
from app import create_app


@pytest.fixture()
def secured(tmp_path):
    app = create_app(offline=True, workdir=tmp_path, admin_token="s3cret")
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture()
def open_client(tmp_path):
    app = create_app(offline=True, workdir=tmp_path, admin_token="")
    app.config["TESTING"] = True
    return app.test_client()


def test_admin_endpoints_need_token(secured):
    assert secured.get("/api/tickets").status_code == 401
    assert secured.post("/api/tickets/TCK-X/resolve").status_code == 401
    assert secured.delete("/api/documents/faq.md").status_code == 401
    assert secured.post("/api/upload", data={}).status_code == 401
    assert secured.get("/api/tickets", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_admin_endpoints_work_with_token(secured):
    h = {"X-Admin-Token": "s3cret"}
    assert secured.get("/api/tickets", headers=h).status_code == 200
    assert secured.post("/api/tickets/TCK-X/resolve", headers=h).status_code == 404  # authorised, ticket unknown


def test_chat_and_status_stay_public(secured):
    assert secured.get("/api/status").get_json()["admin_required"] is True
    assert secured.post("/api/chat", json={"message": "Where is my order SG1002?"}).status_code == 200


def test_requests_without_session_id_do_not_share_memory(open_client):
    open_client.post("/api/chat", json={"message": "check my order status"})
    r = open_client.post("/api/chat", json={"message": "It is SG1001"}).get_json()
    assert r["intent"] == "order"           # still works on its own
    # a second anonymous request must not inherit the first one's history
    a = open_client.post("/api/chat", json={"message": "check my order status"}).get_json()
    assert "order ID" in a["answer"]


def test_order_requires_matching_email_when_enabled(open_client, monkeypatch):
    monkeypatch.setenv("REQUIRE_EMAIL_FOR_ORDERS", "1")
    no_email = open_client.post("/api/chat", json={"session_id": "a", "message": "Status of SG1001"}).get_json()
    assert "couldn't find" in no_email["answer"].lower()
    wrong = open_client.post("/api/chat", json={"session_id": "b", "message": "Status of SG1001",
                                                "email": "attacker@example.com"}).get_json()
    assert "couldn't find" in wrong["answer"].lower()
    ok = open_client.post("/api/chat", json={"session_id": "c", "message": "Status of SG1001",
                                             "email": "ASHA@example.com"}).get_json()
    assert "Delivered" in ok["answer"]


def test_order_email_never_leaks_in_response(open_client):
    r = open_client.post("/api/chat", json={"message": "Status of SG1001"}).get_json()
    assert "@example.com" not in str(r)
