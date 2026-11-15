"""Document ingestion, chunking and retrieval (the RAG layer).

Two interchangeable search back-ends:
  * "gemini" - Gemini embeddings + cosine similarity (used when an API key is set)
  * "tfidf"  - local TF-IDF vectors (offline fallback; also used if the embedding API fails)
"""
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("supportgen.rag")

CHUNK_WORDS, OVERLAP_WORDS = 120, 25
SUPPORTED = {".txt", ".md", ".pdf"}
STOP = set("a an the and or of to in on for is are was were be it this that with as at by from i you my me "
           "do does can how what when where which who your our we if so but not no please long take many much get "
           "make need want would could should will".split())


# ---------- reading & chunking ----------
def read_file(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
    raise ValueError(f"Unsupported file type '{ext}'. Use .txt, .md or .pdf")


def chunk_text(text, source):
    """Split into ~CHUNK_WORDS chunks on paragraph boundaries; a markdown heading always starts a new chunk."""
    pieces, title = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if not pieces and not title and re.fullmatch(r"# [^\n]+", para):  # document title: context, not a chunk
            title = para[2:].strip()
            continue
        words = para.split()
        if len(words) <= CHUNK_WORDS * 1.5:
            pieces.append(para)
        else:  # very long paragraph (typical for PDFs): sliding window
            step = CHUNK_WORDS - OVERLAP_WORDS
            pieces += [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)]
    chunks, cur, n = [], [], 0
    for p in pieces:
        k, heading = len(p.split()), p.startswith("#")
        if cur and (heading or n + k > CHUNK_WORDS):
            chunks.append("\n\n".join(cur))
            last = cur[-1]
            keep = (not heading) and len(last.split()) <= OVERLAP_WORDS  # carry one short paragraph as overlap
            cur, n = ([last], len(last.split())) if keep else ([], 0)
        cur.append(p)
        n += k
    if cur:
        chunks.append("\n\n".join(cur))
    return [{"source": source, "title": title, "text": c} for c in chunks]


# ---------- TF-IDF fallback ----------
def _stem(w):
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not (suf == "s" and w.endswith("ss")):
            w = w[:-len(suf)] + rep
            break
    return w[:-1] if len(w) > 3 and w[-1] == w[-2] else w


def _tokens(text):
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class _Tfidf:
    def fit(self, docs):
        toks = [_tokens(d) for d in docs]
        df = Counter(t for ts in toks for t in set(ts))
        self.vocab = {t: i for i, t in enumerate(df)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in self.vocab])
        return self._vectors(toks)

    def encode(self, texts):
        return self._vectors([_tokens(t) for t in texts])

    def _vectors(self, toks_list):
        m = np.zeros((len(toks_list), len(self.vocab)))
        for r, ts in enumerate(toks_list):
            for t, c in Counter(ts).items():
                if t in self.vocab:
                    m[r, self.vocab[t]] = 1 + math.log(c)
        m *= self.idf
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return m / norm


# ---------- knowledge base ----------
class KnowledgeBase:
    def __init__(self, llm=None):
        self.llm = llm
        self.mode = "gemini" if llm else "tfidf"
        self.chunks, self.vecs = [], None
        self._cache, self._tfidf, self._lock = {}, _Tfidf(), threading.RLock()

    def docs(self):
        counts = Counter(c["source"] for c in self.chunks)
        return [{"name": k, "chunks": v} for k, v in counts.items()]

    def add_document(self, name, text):
        new = chunk_text(text, name)
        if not new:
            raise ValueError("No readable text found in this document")
        with self._lock:
            self.chunks = [c for c in self.chunks if c["source"] != name] + new
            self._reindex()
        return len(new)

    def remove_document(self, name):
        with self._lock:
            before = len(self.chunks)
            self.chunks = [c for c in self.chunks if c["source"] != name]
            self._reindex()
            return before - len(self.chunks)

    def search(self, query, k=4):
        with self._lock:
            if self.vecs is None:
                return []
            q = None
            if self.mode == "gemini":
                try:
                    q = self._unit(self.llm.embed([query], query=True)[0])
                except Exception as e:  # quota / network problem -> degrade gracefully
                    log.warning("Gemini embedding failed (%s); switching to TF-IDF", e)
                    self.mode = "tfidf"
                    self._reindex()
            if q is None:
                q = self._tfidf.encode([query])[0]
            scores = self.vecs @ q
            hits = [{**self.chunks[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]
            if self.mode == "tfidf":  # share of query terms found in the passage (guards against one-word matches)
                qt = set(_tokens(query)) or {""}
                for h in hits:
                    h["coverage"] = len(qt & set(_tokens(h["title"] + " " + h["text"]))) / len(qt)
            return hits

    # -- internals --
    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=float)
        return v / (np.linalg.norm(v) or 1)

    def _reindex(self):
        texts = [f"{c['title']}\n{c['text']}" for c in self.chunks]  # title gives every chunk its document context
        if not texts:
            self.vecs = None
            return
        if self.mode == "gemini":
            try:
                missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
                if missing:
                    for t, v in zip(missing, self.llm.embed(missing)):
                        self._cache[t] = self._unit(v)
                self.vecs = np.vstack([self._cache[t] for t in texts])
                return
            except Exception as e:
                log.warning("Gemini embedding failed (%s); switching to TF-IDF", e)
                self.mode = "tfidf"
        self.vecs = self._tfidf.fit(texts)
