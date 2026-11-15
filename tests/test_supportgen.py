"""Offline HTTP API and RAG unit tests (11 tests, no API key or network required)."""
import io
import pytest
from app import create_app
from rag import chunk_text


@pytest.fixture()
def client(tmp_path):
    app = create_app(offline=True, workdir=tmp_path)
    app.config["TESTING"] = True
    return app.test_client()


def test_status_offline(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    d = r.get_json()
    assert d["mode"] == "offline"
    assert d["search"] == "tfidf"
    assert len(d["documents"]) >= 4


def test_faq_with_source(client):
    r = client.post("/api/chat", json={"message": "What is your 30-day return policy and refund processing time?"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["escalated"] is False
    assert d["confidence"] >= 0.6
    assert any("refund_policy.md" in s["source"] for s in d["sources"])


def test_order_lookup_tool(client):
    r = client.post("/api/chat", json={"message": "Where is my order SG1002?"})
    d = r.get_json()
    assert d["intent"] == "order"
    assert d["escalated"] is False
    assert "SG1002" in d["answer"]
    assert "In transit" in d["answer"]


def test_multiturn_order_id(client):
    r1 = client.post("/api/chat", json={"session_id": "s1", "message": "check my order status"})
    d1 = r1.get_json()
    assert d1["intent"] == "order"
    assert "order ID" in d1["answer"]

    r2 = client.post("/api/chat", json={"session_id": "s1", "message": "It is SG1001"})
    d2 = r2.get_json()
    assert d2["escalated"] is False
    assert "SG1001" in d2["answer"]
    assert "Delivered" in d2["answer"]


def test_unknown_order_id(client):
    r = client.post("/api/chat", json={"message": "Status of SG9999"})
    d = r.get_json()
    assert d["escalated"] is False
    assert "couldn't find" in d["answer"].lower()


def test_escalation_human_request(client):
    r = client.post("/api/chat", json={"message": "Please connect me to a human agent", "email": "user@test.com"})
    d = r.get_json()
    assert d["escalated"] is True
    assert d["ticket_id"].startswith("TCK-")


def test_escalation_unknown_topic(client):
    r = client.post("/api/chat", json={"message": "Do you sell electric scooters?"})
    d = r.get_json()
    assert d["escalated"] is True
    assert d["ticket_id"] is not None


def test_ticket_list_and_resolve(client):
    r = client.post("/api/chat", json={"message": "This is terrible scam service!!!"})
    tid = r.get_json()["ticket_id"]

    tickets = client.get("/api/tickets").get_json()
    assert any(t["id"] == tid and t["status"] == "open" for t in tickets)

    res = client.post(f"/api/tickets/{tid}/resolve")
    assert res.status_code == 200
    assert res.get_json()["status"] == "resolved"


def test_upload_and_query_document(client):
    content = b"# Drone Policy\n\n## SkyPod Drone\nThe SkyPod Drone has a 45-minute flight time and 4K gimbal camera."
    r = client.post("/api/upload", data={"file": (io.BytesIO(content), "drone_guide.md")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["chunks"] >= 1

    q = client.post("/api/chat", json={"message": "What is the flight time of the SkyPod Drone?"}).get_json()
    assert q["escalated"] is False
    assert "45-minute" in q["answer"]

    del_r = client.delete("/api/documents/drone_guide.md")
    assert del_r.status_code == 200
    assert del_r.get_json()["removed_chunks"] >= 1


def test_input_validation(client):
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 1001}).status_code == 400
    assert client.post("/api/upload", data={}).status_code == 400


def test_chunk_text_splits_headings():
    md = "# Main Title\n\n## Section One\nFirst paragraph text.\n\n## Section Two\nSecond paragraph text."
    chunks = chunk_text(md, "sample.md")
    assert len(chunks) == 2
    assert chunks[0]["title"] == "Main Title"
    assert "Section One" in chunks[0]["text"]
    assert "Section Two" in chunks[1]["text"]
