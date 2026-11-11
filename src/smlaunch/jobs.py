"""Consulta, espera e custo estimado de Training Jobs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Stopped"})

# Preço por hora (USD, on-demand, us-east). Sobrescreva em `pricing:` no YAML.
DEFAULT_PRICING = {
    "ml.g4dn.xlarge": 0.736,
    "ml.g4dn.2xlarge": 0.94,
    "ml.g5.xlarge": 1.408,
    "ml.g5.2xlarge": 1.515,
    "ml.p3.2xlarge": 3.825,
}


class JobNotFoundError(LookupError):
    pass


@dataclass
class JobSummary:
    name: str
    status: str
    secondary_status: str
    instance_type: str
    duration_s: int | None
    billable_s: int | None
    estimated_cost_usd: float | None
    failure_reason: str | None
    model_artifacts: str | None

    def render(self) -> str:
        lines = [
            f"Job:            {self.name}",
            f"Status:         {self.status} / {self.secondary_status}",
            f"Instância:      {self.instance_type}",
        ]
        if self.duration_s is not None:
            h, rem = divmod(self.duration_s, 3600)
            m, s = divmod(rem, 60)
            lines.append(f"Duração:        {h}h {m}m {s}s")
        if self.billable_s:
            lines.append(f"Faturável:      {self.billable_s}s")
        if self.estimated_cost_usd is not None:
            lines.append(f"Custo estimado: US$ {self.estimated_cost_usd:.2f}")
        if self.failure_reason:
            lines.append(f"Falha:          {self.failure_reason}")
        return "\n".join(lines)


def estimate_cost(instance_type: str, billable_s: int | None,
                  pricing: dict[str, float] | None = None) -> float | None:
    if not billable_s:
        return None
    table = {**DEFAULT_PRICING, **(pricing or {})}
    hourly = table.get(instance_type)
    if hourly is None:
        return None  # melhor não estimar do que estimar errado
    # BillableTimeInSeconds já é a soma de todas as instâncias
    return billable_s * hourly / 3600


def summarize(desc: dict[str, Any], pricing: dict[str, float] | None = None,
              now: datetime | None = None) -> JobSummary:
    resources = desc.get("ResourceConfig", {})
    instance_type = resources.get("InstanceType", "?")
    started, ended = desc.get("TrainingStartTime"), desc.get("TrainingEndTime")
    duration = None
    if started:
        end = ended or now or datetime.now(timezone.utc)
        duration = int((end - started).total_seconds())
    billable = desc.get("BillableTimeInSeconds")
    return JobSummary(
        name=desc["TrainingJobName"],
        status=desc["TrainingJobStatus"],
        secondary_status=desc.get("SecondaryStatus", "-"),
        instance_type=instance_type,
        duration_s=duration,
        billable_s=billable,
        estimated_cost_usd=estimate_cost(instance_type, billable, pricing),
        failure_reason=desc.get("FailureReason"),
        model_artifacts=desc.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
    )


def describe(sm_client, job_name: str) -> dict[str, Any]:
    try:
        return sm_client.describe_training_job(TrainingJobName=job_name)
    except Exception as exc:  # botocore.ClientError, sem depender de botocore aqui
        if "ValidationException" in str(exc) or "ResourceNotFound" in str(exc):
            raise JobNotFoundError(job_name) from exc
        raise


def wait(sm_client, job_name: str, poll_s: float = 30.0, max_poll_s: float = 300.0,
         on_change: Callable[[str, str], None] | None = None,
         sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Faz polling até o job terminar, com backoff progressivo enquanto nada muda."""
    last = None
    delay = poll_s
    while True:
        desc = describe(sm_client, job_name)
        state = (desc["TrainingJobStatus"], desc.get("SecondaryStatus", ""))
        if state != last:
            if on_change:
                on_change(*state)
            last, delay = state, poll_s
        else:
            delay = min(delay * 1.5, max_poll_s)
        if state[0] in TERMINAL_STATUSES:
            return desc
        sleep(delay)


def list_recent(sm_client, limit: int = 10, name_contains: str | None = None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"SortBy": "CreationTime", "SortOrder": "Descending", "MaxResults": limit}
    if name_contains:
        kwargs["NameContains"] = name_contains
    return sm_client.list_training_jobs(**kwargs).get("TrainingJobSummaries", [])
