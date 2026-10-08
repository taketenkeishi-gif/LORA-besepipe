"""How far down a character's candidate list can be trusted, learned from the user's own decisions.

The automatic grouping puts uncertain images into "X の候補" ordered by a score (lower = more likely X).  The score is only an
ordering; whether its top part is right cannot be measured with the model's own picks (they were chosen by the same model).
The user's edits are the answers: a candidate that now sits in X is right, one that was excluded or put with another character
is wrong.  From those answers the largest score whose candidates are right often enough is found, and the unanswered
candidates above it can be moved in one step.
"""
from __future__ import annotations

import math

TARGET = 0.95       # required share of right answers above the line
MIN_ANSWERS = 30    # answers needed above the line before it is offered
LOWER_BOUND = 0.90  # ...and the 90 % Wilson lower bound of that share must reach this


def wilson_lower(right: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 0.0
    p = right / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / den


def answers(sections: list[dict], state: dict, prefix: str, judged: dict[str, int] | None = None) -> list[dict]:
    """Every candidate image with its score and, if the user decided it, whether it was right (1) or wrong (0).

    sections: crop_cluster.py output sections; state: characters.json.  Main character ids are main index + 1 (_from_result).
    judged: answers given on a check sample (rel -> 1/0); they count where the image was not moved since.
    """
    judged = judged or {}
    where: dict[str, dict] = {}
    for c in state["characters"]:
        for rel in c["images"]:
            where[rel] = c
    excluded = set(state.get("excluded", []))
    out = []
    for s in sections:
        if s.get("section") != "candidates" or "scores" not in s:
            continue
        parent = s["main"] + 1
        for f, sc in zip(s["files"], s["scores"]):
            rel = prefix + f
            now = where.get(rel)
            if rel in excluded:
                label = 0
            elif now is None:
                label = None
            elif now["id"] == parent:
                label = 1
            elif now.get("pending") and now.get("section") in ("candidates", "multi", "unknown"):
                label = judged.get(rel)  # still waiting (answered on a check sample, or not at all)
            else:
                label = 0  # put with another character
            out.append({"rel": rel, "parent": parent, "score": sc[0], "label": label, "in": now["id"] if now else None})
    return out


LOCAL = 15         # the last LOCAL answers before the line must also be right at LOCAL_TARGET (the tail must not be worse)
LOCAL_TARGET = 0.85


def line(rows: list[dict]) -> dict:
    """The largest score up to which the answered candidates are right often enough (or none yet).
    Answers spread over the score range (a check sample) let the line reach unanswered candidates between them."""
    answered = sorted((r for r in rows if r["label"] is not None), key=lambda r: r["score"])
    right = wrong = 0
    best = None
    for k, r in enumerate(answered, 1):
        right += r["label"]
        wrong += 1 - r["label"]
        tail = answered[max(0, k - LOCAL):k]
        if k >= MIN_ANSWERS and right / k >= TARGET and wilson_lower(right, k) >= LOWER_BOUND and \
                sum(t["label"] for t in tail) / len(tail) >= LOCAL_TARGET:
            best = {"score": r["score"], "answers": k, "right": right, "precision": round(right / k, 3)}
    return {"answered": len(answered), "right": sum(r["label"] for r in answered), "wrong": sum(1 - r["label"] for r in answered),
            "line": best}


def sample(rows: list[dict], n: int, seed: int = 0) -> list[str]:
    """n unanswered candidates spread evenly over the score range (best to worst), for a quick check."""
    import random

    open_rows = sorted((r for r in rows if r["label"] is None), key=lambda r: r["score"])
    if len(open_rows) <= n:
        return [r["rel"] for r in open_rows]
    rng = random.Random(seed)
    step = len(open_rows) / n
    return [open_rows[min(len(open_rows) - 1, int(k * step + rng.random() * step))]["rel"] for k in range(n)]


def above(rows: list[dict], cut: float) -> list[dict]:
    """Undecided candidates still in their candidate pile whose score is within the line."""
    return [r for r in rows if r["label"] is None and r["score"] <= cut]
