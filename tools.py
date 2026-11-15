"""Tools the agent can use, plus the escalation ticket store.

get_order_details is exposed to Gemini as a callable tool. Ticket creation is deliberately NOT given to the
model: escalation is a deterministic policy decision made by the pipeline (see agent.py).
"""
import contextvars
import json
import os
import threading
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path


def _ago(days):
    return (date.today() - timedelta(days=days)).isoformat()


# Mock order database (replace with your real order API / DB)
ORDERS = {
    "SG1001": {"email": "asha@example.com", "status": "Delivered", "items": ["SphereBuds Pro (Black)"], "total_inr": 7999,
               "ordered_on": _ago(14), "delivered_on": _ago(10), "carrier": "BlueDart", "tracking": "BD778123"},
    "SG1002": {"email": "ravi@example.com", "status": "In transit", "items": ["SphereBuds Pro (White)", "USB-C cable"], "total_inr": 8498,
               "ordered_on": _ago(3), "expected_delivery": _ago(-2), "carrier": "Delhivery", "tracking": "DL445210"},
    "SG1003": {"email": "asha@example.com", "status": "Delivered", "items": ["SphereWatch Lite"], "total_inr": 4999,
               "ordered_on": _ago(50), "delivered_on": _ago(45), "carrier": "Ecom Express", "tracking": "EE990017"},
    "SG1004": {"email": "meera@example.com", "status": "Processing", "items": ["SphereBuds Pro (Black)"], "total_inr": 7999,
               "ordered_on": _ago(0), "carrier": None, "tracking": None},
}

# Agent steps and tool calls share one trace list per request (shown in the UI's "Agent steps" panel)
_trace = contextvars.ContextVar("trace", default=None)
# Email the customer typed in the UI for this request; used to verify order ownership
_email = contextvars.ContextVar("email", default="")


def set_email(email):
    _email.set((email or "").strip().lower())


def set_trace(lst):
    _trace.set(lst)


def _log(name, args, result):
    t = _trace.get()
    if t is not None:
        t.append({"step": "tool call", "result": result,
                  "detail": f"{name}({args}) -> found={result.get('found')}, status={result.get('status')}"})


def get_order_details(order_id: str) -> dict:
    """Look up a customer order by its order ID (format: 'SG' followed by 4 digits, e.g. SG1001).
    Returns status, items, order date, delivery date, carrier, tracking number and days_since_delivery.
    Use for any question about order status, tracking, delivery dates or return eligibility."""
    oid = (order_id or "").strip().upper()
    order = ORDERS.get(oid)
    # REQUIRE_EMAIL_FOR_ORDERS=1: only show an order to the email that placed it. A mismatch looks exactly like
    # "not found" so order IDs cannot be probed. Off by default so the demo works without typing an email.
    if order and os.getenv("REQUIRE_EMAIL_FOR_ORDERS", "0") == "1" and _email.get() != order["email"].lower():
        order = None
    if not order:
        result = {"found": False, "error": f"No order found with ID '{oid}' for this customer"}
    else:
        public = {k: v for k, v in order.items() if k != "email"}  # never expose the email to the model/UI
        result = {"found": True, "order_id": oid, **public}
        if order.get("delivered_on"):
            result["days_since_delivery"] = (date.today() - date.fromisoformat(order["delivered_on"])).days
    _log("get_order_details", {"order_id": oid}, result)
    return result


class TicketStore:
    """Escalation tickets persisted to a JSON file (swap for a helpdesk API such as Zendesk in production)."""

    def __init__(self, path):
        self.path, self._lock = Path(path), threading.Lock()
        self.items = []
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.items = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=2), encoding="utf-8")

    def create(self, question, reason, priority="normal", email="", session_id="", transcript=None):
        ticket = {"id": "TCK-" + uuid.uuid4().hex[:6].upper(), "status": "open", "priority": priority,
                  "created": datetime.now().isoformat(timespec="seconds"), "question": question,
                  "reason": reason, "email": email, "session_id": session_id, "transcript": transcript or []}
        with self._lock:
            self.items.append(ticket)
            self._save()
        return ticket

    def list(self):
        return list(reversed(self.items))

    def resolve(self, ticket_id):
        with self._lock:
            for t in self.items:
                if t["id"] == ticket_id:
                    t["status"] = "resolved"
                    self._save()
                    return t
        return None
