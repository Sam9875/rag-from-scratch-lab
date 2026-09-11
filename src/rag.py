"""Minimal RAG: hashing embeddings so it runs without a model download."""

from __future__ import annotations

import hashlib
import math
import re

DOCS = [
    "Column news ranks stories for new users with a two-tower retriever and a popularity prior.",
    "MIND Large is a public stand-in for news impression logs when production traffic is private.",
    "A tenant-bias audit ran 7800 LLM calls on Turin rental listings.",
    "Stellantis breakdown risk uses CatBoost on more than a million service logs.",
]


def chunk(text: str, size: int = 12) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def embed(text: str, dim: int = 32) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] -= 0.3
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve(query: str, k: int = 2) -> list[tuple[float, str]]:
    pieces = []
    for doc in DOCS:
        pieces.extend(chunk(doc))
    qv = embed(query)
    ranked = sorted(((cosine(qv, embed(p)), p) for p in pieces), reverse=True)
    return ranked[:k]


def generate(query: str) -> str:
    hits = retrieve(query)
    conf = hits[0][0] if hits else 0.0
    if conf < 0.15:
        return f"ABSTAIN (confidence={conf:.2f}). No grounded chunk."
    ctx = " | ".join(p for _, p in hits)
    return f"Q: {query}\ncontext: {ctx}\nconfidence: {conf:.2f}"


if __name__ == "__main__":
    print(generate("How do you rank news for a brand-new user?"))
    print("---")
    print(generate("What is the capital of Atlantis?"))
