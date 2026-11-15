"""Thin wrapper around the Google Gemini API (google-genai SDK)."""
import json
import os
import re

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")


class Gemini:
    def __init__(self, api_key):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt, system=None, tools=None, json_mode=False, temperature=0.2):
        """tools = plain Python functions; the SDK turns their signatures/docstrings into Gemini function
        declarations and runs the call -> execute -> respond loop automatically (agentic tool calling)."""
        model = os.getenv("GEMINI_MODEL", MODEL)
        cfg = {"system_instruction": system, "temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if tools:
            cfg["tools"] = tools
        resp = self.client.models.generate_content(
            model=model, contents=prompt, config=self.types.GenerateContentConfig(**cfg))
        return resp.text or ""

    def embed(self, texts, query=False):
        embed_model = os.getenv("GEMINI_EMBED_MODEL", EMBED_MODEL)
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        out = []
        for i in range(0, len(texts), 50):  # API batch limit
            r = self.client.models.embed_content(
                model=embed_model, contents=texts[i:i + 50],
                config=self.types.EmbedContentConfig(task_type=task))
            out += [e.values for e in r.embeddings]
        return out


def parse_json(text, default=None):
    """Parse model output as JSON, tolerating ```json fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return default
