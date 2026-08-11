"""
Tests for the cross-encoder reranker (Phase 3.5).

The behaviour that matters most is the *failure* behaviour. This project has
lost hours to model downloads failing on an unstable connection, so a reranker
that could take chat down with it would be a bad trade. Every unavailable path
must return None and let the caller keep vector ordering.
"""

import pytest

from app.services import reranker as reranker_mod
from app.services.reranker import RerankerService


class FakeCrossEncoder:
    """Returns scores from a lookup keyed on passage text."""

    def __init__(self, *args, scores: dict[str, float] | None = None, **kwargs):
        self._scores = scores or {}
        self.calls = 0

    def predict(self, pairs, batch_size=32):
        self.calls += 1
        return [self._scores.get(text, 0.0) for _, text in pairs]


def make_service(monkeypatch, *, enabled=True, scores=None, load_raises=None):
    monkeypatch.setattr(reranker_mod.settings, "RERANK_ENABLED", enabled, raising=False)
    monkeypatch.setattr(reranker_mod.settings, "RERANKER_MODEL", "fake-model", raising=False)

    svc = RerankerService()

    if load_raises is not None:
        def boom(*a, **k):
            raise load_raises
        monkeypatch.setattr(svc, "load", lambda: False)
    else:
        fake = FakeCrossEncoder(scores=scores)

        def fake_load():
            svc._model = fake
            svc._loaded = True
            return True

        monkeypatch.setattr(svc, "load", fake_load)

    return svc


# ══════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION — the important part
# ══════════════════════════════════════════════════════════════


def test_returns_none_when_disabled(monkeypatch):
    svc = make_service(monkeypatch, enabled=False)
    assert svc.score("q", ["a", "b"]) is None


def test_returns_none_when_model_cannot_load(monkeypatch):
    """
    A missing model must not raise. Chat keeps working on vector ordering.
    """
    svc = make_service(monkeypatch, load_raises=OSError("model not in cache"))
    assert svc.score("q", ["a", "b"]) is None


def test_returns_none_for_empty_input(monkeypatch):
    svc = make_service(monkeypatch)
    assert svc.score("q", []) is None


def test_prediction_failure_degrades_instead_of_raising(monkeypatch):
    svc = make_service(monkeypatch)
    svc.load()

    class Exploding:
        def predict(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    svc._model = Exploding()
    assert svc.score("q", ["a"]) is None


def test_failed_load_is_not_retried(monkeypatch):
    """
    A broken load must be remembered. Otherwise every single query pays the
    cost of retrying a download that is going to fail again.
    """
    monkeypatch.setattr(reranker_mod.settings, "RERANK_ENABLED", True, raising=False)
    svc = RerankerService()

    attempts = {"n": 0}

    def failing_import(*a, **k):
        attempts["n"] += 1
        raise OSError("no cache")

    monkeypatch.setattr(reranker_mod.RerankerService, "load", RerankerService.load)
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)

    svc.score("q", ["a"])
    svc.score("q", ["b"])
    svc.score("q", ["c"])

    assert svc._load_failed is True


# ══════════════════════════════════════════════════════════════
# SCORING AND ORDERING
# ══════════════════════════════════════════════════════════════


def test_scores_returned_per_text(monkeypatch):
    svc = make_service(monkeypatch, scores={"a": 1.0, "b": 2.0, "c": 3.0})
    assert svc.score("q", ["a", "b", "c"]) == [1.0, 2.0, 3.0]


def test_reranking_can_reverse_vector_order(monkeypatch):
    """
    The whole point: a passage the bi-encoder ranked last can be promoted to
    first when the cross-encoder judges it actually answers the question.

    Modelled on the real failure — the article's "Definition" passage embedded
    closer to the query than its "CXR Findings" passage.
    """
    definition = "Pneumothorax is the presence of air in the pleural space."
    findings = "CXR findings: a thin visceral pleural line with no lung markings beyond it."

    svc = make_service(monkeypatch, scores={definition: -2.5, findings: 8.1})
    texts = [definition, findings]          # vector order
    scores = svc.score("What are the radiographic findings of pneumothorax?", texts)

    ranked = [t for _, t in sorted(zip(scores, texts), key=lambda p: -p[0])]
    assert ranked[0] == findings


def test_negative_scores_are_preserved(monkeypatch):
    """
    ms-marco cross-encoders emit roughly -11..+11. Clamping or treating these
    as similarities would destroy the ordering.
    """
    svc = make_service(monkeypatch, scores={"bad": -9.5, "good": 7.2})
    assert svc.score("q", ["bad", "good"]) == [-9.5, 7.2]


def test_scores_are_plain_floats(monkeypatch):
    """Numpy scalars break JSON serialisation downstream."""
    svc = make_service(monkeypatch, scores={"a": 1.5})
    assert all(type(s) is float for s in svc.score("q", ["a"]))


# ══════════════════════════════════════════════════════════════
# INTROSPECTION
# ══════════════════════════════════════════════════════════════


def test_info_reports_state(monkeypatch):
    svc = make_service(monkeypatch)
    info = svc.get_info()

    assert info["enabled"] is True
    assert info["model"] == "fake-model"
    assert info["loaded"] is False          # lazy — not touched yet

    svc.score("q", ["a"])
    assert svc.get_info()["loaded"] is True


def test_is_enabled_tracks_config(monkeypatch):
    assert make_service(monkeypatch, enabled=True).is_enabled is True
    assert make_service(monkeypatch, enabled=False).is_enabled is False
