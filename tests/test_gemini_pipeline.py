"""Gemini-mode pipeline unit tests with a fake LLM (5 tests, no API key or network required)."""
import json
import numpy as np
import tools
from agent import Agent
from rag import KnowledgeBase


class FakeLLM:
    def __init__(self, analyze_out=None, draft_out="Grounded answer [1].", verify_out=None,
                 fail_generate=False, fail_embed=False):
        self.analyze_out = analyze_out or {"intent": "faq", "sentiment": "neutral",
                                           "standalone_question": "return policy"}
        self.draft_out = draft_out
        self.verify_out = verify_out or {"grounded": True, "confidence": 0.92, "issue": "none"}
        self.fail_generate = fail_generate
        self.fail_embed = fail_embed

    def generate(self, prompt, system=None, tools=None, json_mode=False, temperature=0.2):
        if self.fail_generate:
            raise RuntimeError("Simulated Gemini API quota exceeded")
        if json_mode and "Classify the latest customer message" in prompt:
            return json.dumps(self.analyze_out)
        if json_mode and "strict QA auditor" in prompt:
            return json.dumps(self.verify_out)
        return self.draft_out

    def embed(self, texts, query=False):
        if self.fail_embed:
            raise RuntimeError("Simulated embedding endpoint error")
        return [np.ones(8, dtype=float).tolist() for _ in texts]


def _make_agent(tmp_path, llm):
    kb = KnowledgeBase(llm)
    kb.add_document("refund_policy.md", "# Refund Policy\n\n## 30-Day Returns\nItems can be returned within 30 days.")
    return Agent(kb, llm, tools.TicketStore(tmp_path / "tickets.json"))


def test_gemini_confident_answer(tmp_path):
    agent = _make_agent(tmp_path, FakeLLM())
    res = agent.handle("s1", "What is your return policy?")
    assert res["mode"] == "gemini"
    assert res["escalated"] is False
    assert res["confidence"] == 0.92
    assert "Grounded answer" in res["answer"]


def test_gemini_low_confidence_escalates(tmp_path):
    llm = FakeLLM(verify_out={"grounded": True, "confidence": 0.35, "issue": "incomplete context"})
    agent = _make_agent(tmp_path, llm)
    res = agent.handle("s1", "Do you ship to Antarctica?")
    assert res["escalated"] is True
    assert res["ticket_id"].startswith("TCK-")


def test_gemini_ungrounded_answer_capped(tmp_path):
    llm = FakeLLM(verify_out={"grounded": False, "confidence": 0.95, "issue": "hallucinated policy"})
    agent = _make_agent(tmp_path, llm)
    res = agent.handle("s1", "What is your return policy?")
    assert res["escalated"] is True
    assert res["confidence"] == 0.0


def test_gemini_api_outage_escalates_gracefully(tmp_path):
    llm = FakeLLM(fail_generate=True)
    agent = _make_agent(tmp_path, llm)
    res = agent.handle("s1", "What is your return policy?")
    assert res["escalated"] is True
    assert any(s["step"] == "error" for s in res["steps"])


def test_embedding_failure_falls_back_to_tfidf(tmp_path):
    llm = FakeLLM(fail_embed=True)
    kb = KnowledgeBase(llm)
    kb.add_document("faq.md", "# FAQ\n\n## Warranty\nAll products include a 1-year manufacturer warranty.")
    assert kb.mode == "tfidf"
    hits = kb.search("1-year manufacturer warranty")
    assert len(hits) >= 1
    assert hits[0]["score"] > 0.1
