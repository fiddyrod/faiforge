import uuid
import time
from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EvalStoreBackend(Protocol):
    """Interface for eval result persistence.

    The default implementation (InMemoryEvalStore) keeps everything in memory.
    To swap in a persistent backend (SQLite, Postgres, etc.) in a fork,
    implement this protocol and replace the instantiation in create_app().
    """

    def append_feedback(self, record: dict) -> None: ...
    def get_feedbacks(self, limit: int = 100, offset: int = 0) -> List[dict]: ...
    def append_ab_result(self, record: dict) -> None: ...
    def get_ab_results(self, experiment_id: str) -> List[dict]: ...


class InMemoryEvalStore:
    """Default in-memory eval store.

    Thread-safe for FastAPI's single-threaded async event loop.
    Not suitable for multi-worker deployments — swap to a shared backend
    (Redis, SQLite, Postgres) when scaling beyond one worker.

    Fork extension point:
        Implement EvalStoreBackend and pass to create_app() via config or DI.
    """

    def __init__(self):
        self._feedbacks: List[dict] = []
        self._ab_results: List[dict] = []

    def append_feedback(self, record: dict) -> None:
        record.setdefault("id", str(uuid.uuid4()))
        record.setdefault("created_at", time.time())
        self._feedbacks.append(record)

    def get_feedbacks(self, limit: int = 100, offset: int = 0) -> List[dict]:
        return self._feedbacks[offset: offset + limit]

    def total_feedbacks(self) -> int:
        return len(self._feedbacks)

    def append_ab_result(self, record: dict) -> None:
        record.setdefault("created_at", time.time())
        self._ab_results.append(record)

    def get_ab_results(self, experiment_id: str) -> List[dict]:
        return [r for r in self._ab_results if r.get("experiment_id") == experiment_id]
