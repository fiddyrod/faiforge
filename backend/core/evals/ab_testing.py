import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .store import EvalStoreBackend
from ..observability import get_logger, log_with_context


@dataclass
class Variant:
    id: str
    system_prompt: str
    weight: float = 0.5  # used for weighted random routing


@dataclass
class Experiment:
    id: str
    variants: List[Variant]
    routing: str = "random"  # "random" | "round_robin"


class ABRouter:
    """Routes requests across prompt variants and records per-variant metrics.

    Usage:
        router = ABRouter(store=eval_store)
        router.register_experiment(Experiment(
            id="greeting-test",
            variants=[
                Variant(id="formal", system_prompt="You are a formal assistant."),
                Variant(id="casual", system_prompt="You are a casual assistant."),
            ],
            routing="round_robin",
        ))

        # In your request handler:
        variant = router.select_variant("greeting-test")
        # inject variant.system_prompt into messages ...
        router.record_result("greeting-test", variant.id, {
            "latency_ms": 123.4,
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.0001,
        })
    """

    def __init__(self, store: EvalStoreBackend):
        self._store = store
        self._experiments: Dict[str, Experiment] = {}
        self._counters: Dict[str, int] = {}  # round-robin counters per experiment
        self.logger = get_logger()

    def register_experiment(self, experiment: Experiment) -> None:
        if not experiment.variants:
            raise ValueError(f"Experiment '{experiment.id}' must have at least one variant.")
        self._experiments[experiment.id] = experiment
        self._counters[experiment.id] = 0
        log_with_context(
            self.logger, "info",
            f"A/B experiment registered: {experiment.id}",
            event="ab_experiment_registered",
            experiment_id=experiment.id,
            variants=[v.id for v in experiment.variants],
            routing=experiment.routing,
        )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[dict]:
        return [
            {
                "id": exp.id,
                "routing": exp.routing,
                "variants": [{"id": v.id, "weight": v.weight} for v in exp.variants],
            }
            for exp in self._experiments.values()
        ]

    def select_variant(self, experiment_id: str) -> Variant:
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found.")

        if exp.routing == "round_robin":
            idx = self._counters[experiment_id] % len(exp.variants)
            self._counters[experiment_id] += 1
            return exp.variants[idx]

        # weighted random (default)
        total = sum(v.weight for v in exp.variants)
        r = random.uniform(0, total)
        cumulative = 0.0
        for v in exp.variants:
            cumulative += v.weight
            if r <= cumulative:
                return v
        return exp.variants[-1]

    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        metrics: dict,
    ) -> None:
        self._store.append_ab_result({
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            **metrics,
        })

    def get_stats(self, experiment_id: str) -> dict:
        """Return aggregated per-variant stats for an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found.")

        results = self._store.get_ab_results(experiment_id)

        variant_stats: Dict[str, dict] = {
            v.id: {
                "id": v.id,
                "requests": 0,
                "total_latency_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
            }
            for v in exp.variants
        }

        for r in results:
            vid = r.get("variant_id")
            if vid not in variant_stats:
                continue
            s = variant_stats[vid]
            s["requests"] += 1
            s["total_latency_ms"] += r.get("latency_ms", 0.0)
            s["total_input_tokens"] += r.get("input_tokens", 0)
            s["total_output_tokens"] += r.get("output_tokens", 0)
            s["total_cost_usd"] += r.get("cost_usd", 0.0)

        aggregated = []
        for vid, s in variant_stats.items():
            n = s["requests"]
            aggregated.append({
                "id": vid,
                "requests": n,
                "avg_latency_ms": round(s["total_latency_ms"] / n, 1) if n else 0.0,
                "avg_input_tokens": round(s["total_input_tokens"] / n, 1) if n else 0.0,
                "avg_output_tokens": round(s["total_output_tokens"] / n, 1) if n else 0.0,
                "avg_cost_usd": round(s["total_cost_usd"] / n, 6) if n else 0.0,
                "total_cost_usd": round(s["total_cost_usd"], 6),
            })

        return {
            "experiment_id": experiment_id,
            "routing": exp.routing,
            "total_requests": len(results),
            "variants": aggregated,
        }
