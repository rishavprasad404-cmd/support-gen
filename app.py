"""SupportGen - Flask backend.   Run:  python app.py   then open http://localhost:5000"""
import hmac
import logging
import os
import uuid
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import tools
from agent import THRESHOLD, Agent
from rag import SUPPORTED, KnowledgeBase, read_file

PLACEHOLDER_KEYS = {"", "your-key-here", "your_api_key_here", "none", "null"}


def create_app(offline=False, workdir=None, admin_token=None):
    """offline=True ignores any API key (used by tests); workdir holds uploads + tickets;
    admin_token overrides ADMIN_TOKEN (when set, ticket/upload/delete endpoints need header X-Admin-Token)."""
    work = Path(workdir or BASE / "data")
    uploads = work / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    raw_key = None if offline else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    key = None if not raw_key or raw_key.strip().lower() in PLACEHOLDER_KEYS else raw_key.strip()
    llm = None
    if key:
        try:
            from llm import Gemini
            llm = Gemini(key)
        except Exception as e:
            logging.warning("Gemini unavailable (%s) - running in offline mode", e)

    kb = KnowledgeBase(llm)
    sample_dir = BASE / "data" / "sample_docs"
    sample_files = list(sample_dir.glob("*")) if sample_dir.exists() else []
    for f in sorted([*sample_files, *uploads.glob("*")]):
        if f.suffix.lower() in SUPPORTED:
            kb.add_document(f.name, read_file(f))
    agent = Agent(kb, llm, tools.TicketStore(work / "tickets.json"))

    token = (admin_token if admin_token is not None else os.getenv("ADMIN_TOKEN", "")).strip()
    if not token:
        logging.warning("ADMIN_TOKEN is not set: ticket, upload and delete endpoints are open (fine for a local demo only)")

    def admin_only(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if token and not hmac.compare_digest(request.headers.get("X-Admin-Token", ""), token):
                return jsonify(error="Admin token required"), 401
            return fn(*a, **kw)
        return wrapper

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(mode=agent.mode, search=kb.mode, threshold=THRESHOLD, documents=kb.docs(), admin_required=bool(token))

    @app.post("/api/chat")
    def chat():
        d = request.get_json(silent=True) or {}
        msg = (d.get("message") or "").strip()
        if not msg:
            return jsonify(error="Message is empty"), 400
        if len(msg) > 1000:
            return jsonify(error="Message too long (max 1000 characters)"), 400
        # no shared "default" session: a client without a session_id gets a private one-off session
        sid = str(d.get("session_id") or "")[:64] or "anon-" + uuid.uuid4().hex
        return jsonify(agent.handle(sid, msg, str(d.get("email") or "")[:120]))

    @app.post("/api/upload")
    @admin_only
    def upload():
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify(error="No file provided"), 400
        name = secure_filename(f.filename)
        if Path(name).suffix.lower() not in SUPPORTED:
            return jsonify(error="Unsupported file type. Upload a .txt, .md or .pdf file."), 400
        path = uploads / name
        f.save(path)
        try:
            n = kb.add_document(name, read_file(path))
        except Exception as e:
            path.unlink(missing_ok=True)
            return jsonify(error=f"Could not read document: {e}"), 400
        return jsonify(name=name, chunks=n)

    @app.get("/api/documents")
    def documents():
        return jsonify(kb.docs())

    @app.delete("/api/documents/<name>")
    @admin_only
    def delete_document(name):
        removed = kb.remove_document(name)
        (uploads / secure_filename(name)).unlink(missing_ok=True)
        return jsonify(removed_chunks=removed)

    @app.get("/api/tickets")
    @admin_only
    def list_tickets():
        return jsonify(agent.tickets.list())

    @app.post("/api/tickets/<ticket_id>/resolve")
    @admin_only
    def resolve_ticket(ticket_id):
        t = agent.tickets.resolve(ticket_id)
        return (jsonify(t), 200) if t else (jsonify(error="Ticket not found"), 404)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File too large (max 10 MB)"), 413

    return app

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)