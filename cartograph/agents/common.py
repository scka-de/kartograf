from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime

from cartograph.core import storage
from cartograph.core.models import Decision

_DECISION_OBSERVER: Callable[[str, Decision], None] | None = None


def next_step(audit_id: str) -> int:
    try:
        return len(storage.load_coverage_report(audit_id).decisions) + 1
    except Exception:
        return 1


def log_decision(audit_id: str, action: str, inputs: dict, outputs: dict, reason: str) -> Decision:
    decision = Decision(
        step=next_step(audit_id),
        timestamp=datetime.utcnow(),
        action=action,
        inputs=inputs,
        outputs=outputs,
        reason=reason,
    )
    storage.save_decision(audit_id, decision)
    if _DECISION_OBSERVER is not None:
        _DECISION_OBSERVER(audit_id, decision)
    return decision


@contextmanager
def observe_decisions(callback: Callable[[str, Decision], None]) -> Iterator[None]:
    global _DECISION_OBSERVER
    previous = _DECISION_OBSERVER
    _DECISION_OBSERVER = callback
    try:
        yield
    finally:
        _DECISION_OBSERVER = previous
