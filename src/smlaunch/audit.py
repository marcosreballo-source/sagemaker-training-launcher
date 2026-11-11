"""Auditoria de custo: recursos ainda ativos e gasto do mês (Cost Explorer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class ActiveResources:
    training_jobs: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    notebooks: list[str] = field(default_factory=list)
    ec2_instances: list[str] = field(default_factory=list)

    @property
    def all_clear(self) -> bool:
        return not (self.training_jobs or self.endpoints or self.notebooks or self.ec2_instances)

    def render(self) -> str:
        def block(title: str, items: list[str]) -> str:
            return f"{title}: {len(items)}" + "".join(f"\n  - {i}" for i in items)
        parts = [
            block("Training jobs em execução", self.training_jobs),
            block("Endpoints ativos", self.endpoints),
            block("Notebook instances ligadas", self.notebooks),
            block("EC2 em execução", self.ec2_instances),
            "OK: nenhum recurso gerando custo." if self.all_clear
            else "ATENÇÃO: há recursos ativos gerando custo.",
        ]
        return "\n".join(parts)


def find_active_resources(session) -> ActiveResources:
    sm = session.client("sagemaker")
    res = ActiveResources()
    for page in sm.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        res.training_jobs += [j["TrainingJobName"] for j in page["TrainingJobSummaries"]]
    for page in sm.get_paginator("list_endpoints").paginate():
        res.endpoints += [f"{e['EndpointName']} ({e['EndpointStatus']})" for e in page["Endpoints"]]
    for page in sm.get_paginator("list_notebook_instances").paginate():
        res.notebooks += [n["NotebookInstanceName"] for n in page["NotebookInstances"]
                          if n["NotebookInstanceStatus"] not in ("Stopped", "Deleting", "Failed")]
    ec2 = session.client("ec2")
    flt = [{"Name": "instance-state-name", "Values": ["pending", "running"]}]
    for page in ec2.get_paginator("describe_instances").paginate(Filters=flt):
        for r in page["Reservations"]:
            res.ec2_instances += [f"{i['InstanceId']} ({i['InstanceType']})" for i in r["Instances"]]
    return res


@dataclass
class MonthCost:
    start: str
    end: str
    total_usd: float
    by_service: list[tuple[str, float]]
    daily: list[tuple[str, float]]
    by_tag: list[tuple[str, float]]
    forecast_usd: float | None

    def render(self) -> str:
        lines = [f"Gasto de {self.start} a {self.end}: US$ {self.total_usd:.2f}"]
        if self.forecast_usd is not None:
            lines.append(f"Previsão do mês:   US$ {self.forecast_usd:.2f}")
        if self.by_tag:
            lines.append("Por tag:")
            lines += [f"  US$ {v:>9.2f}  {k}" for k, v in self.by_tag]
        lines.append("Por serviço:")
        lines += [f"  US$ {v:>9.2f}  {k}" for k, v in self.by_service] or ["  (sem gastos)"]
        lines.append("Últimos dias:")
        lines += [f"  {d}  US$ {v:>8.2f}" for d, v in self.daily]
        return "\n".join(lines)


def month_cost(session, today: date | None = None, tag_key: str | None = None,
               last_days: int = 7) -> MonthCost:
    """Gasto do mês corrente. Cost Explorer só responde em us-east-1."""
    ce = session.client("ce", region_name="us-east-1")
    today = today or date.today()
    start = today.replace(day=1).isoformat()
    end = (today + timedelta(days=1)).isoformat()
    period = {"Start": start, "End": end}

    def grouped(key_type: str, key: str) -> list[tuple[str, float]]:
        resp = ce.get_cost_and_usage(TimePeriod=period, Granularity="MONTHLY",
                                     Metrics=["UnblendedCost"],
                                     GroupBy=[{"Type": key_type, "Key": key}])
        out: dict[str, float] = {}
        for r in resp["ResultsByTime"]:
            for g in r["Groups"]:
                out[g["Keys"][0]] = out.get(g["Keys"][0], 0.0) + float(g["Metrics"]["UnblendedCost"]["Amount"])
        return sorted(((k, v) for k, v in out.items() if v > 0.005), key=lambda kv: -kv[1])

    by_service = grouped("DIMENSION", "SERVICE")
    by_tag = grouped("TAG", tag_key) if tag_key else []

    daily_start = max(today.replace(day=1), today - timedelta(days=last_days)).isoformat()
    daily_resp = ce.get_cost_and_usage(TimePeriod={"Start": daily_start, "End": end},
                                       Granularity="DAILY", Metrics=["UnblendedCost"])
    daily = [(r["TimePeriod"]["Start"], float(r["Total"]["UnblendedCost"]["Amount"]))
             for r in daily_resp["ResultsByTime"]]

    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()
    forecast = None
    try:
        fc = ce.get_cost_forecast(TimePeriod={"Start": end, "End": next_month},
                                  Metric="UNBLENDED_COST", Granularity="MONTHLY")
        forecast = float(fc["Total"]["Amount"])
    except Exception:
        pass  # o Cost Explorer recusa previsão sem histórico suficiente

    return MonthCost(start=start, end=end, total_usd=sum(v for _, v in by_service),
                     by_service=by_service, daily=daily, by_tag=by_tag, forecast_usd=forecast)
