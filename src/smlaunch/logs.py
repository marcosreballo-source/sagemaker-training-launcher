"""Download dos logs do CloudWatch de um Training Job."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

LOG_GROUP = "/aws/sagemaker/TrainingJobs"


def fetch_events(logs_client, job_name: str, log_group: str = LOG_GROUP) -> list[dict]:
    """Lê todos os eventos de todos os streams do job (um por instância), em ordem de tempo."""
    streams: list[str] = []
    kwargs = {"logGroupName": log_group, "logStreamNamePrefix": f"{job_name}/"}
    while True:
        resp = logs_client.describe_log_streams(**kwargs)
        streams += [s["logStreamName"] for s in resp.get("logStreams", [])]
        if not resp.get("nextToken"):
            break
        kwargs["nextToken"] = resp["nextToken"]

    events: list[dict] = []
    for stream in streams:
        kw = {"logGroupName": log_group, "logStreamName": stream, "startFromHead": True}
        while True:
            resp = logs_client.get_log_events(**kw)
            events += resp.get("events", [])
            token = resp.get("nextForwardToken")
            if not resp.get("events") or token == kw.get("nextToken"):
                break  # o CloudWatch devolve o mesmo token quando acabou
            kw["nextToken"] = token
    events.sort(key=lambda e: e["timestamp"])
    return events


def write_log(events: list[dict], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for e in events:
            ts = datetime.fromtimestamp(e["timestamp"] / 1000, tz=timezone.utc)
            fh.write(f"[{ts:%Y-%m-%d %H:%M:%S}] {e['message'].rstrip()}\n")
    return dest
