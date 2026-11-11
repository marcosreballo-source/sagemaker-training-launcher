"""Montagem e lançamento do Training Job, e coleta dos resultados."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from smlaunch import artifacts, jobs, logs
from smlaunch.config import LaunchConfig
from smlaunch.entry import COMMAND_KEY, OUTPUTS_KEY
from smlaunch.naming import build_job_name, build_tags

ENTRY_FILENAME = "_smlaunch_entry.py"
LAST_JOB_FILE = ".smlaunch_last_job"


@dataclass
class LaunchPlan:
    """Tudo o que será enviado ao SageMaker, calculado sem tocar na AWS (base do --dry-run)."""
    base_job_name: str
    command: list[str]
    hyperparameters: dict[str, str]
    tags: list[dict[str, str]]
    estimator_kwargs: dict[str, Any]

    def render(self) -> str:
        kw = self.estimator_kwargs
        tags = ", ".join(f"{t['Key']}={t['Value']}" for t in self.tags) or "-"
        return "\n".join([
            f"Job (base):  {self.base_job_name}",
            f"Comando:     {' '.join(self.command)}",
            f"Instância:   {kw['instance_count']} x {kw['instance_type']}"
            + ("  [spot]" if kw.get("use_spot_instances") else ""),
            f"Max run:     {kw['max_run'] / 3600:.1f} h",
            f"Framework:   {kw['framework']} {kw['framework_version']} ({kw['py_version']})",
            f"Source dir:  {kw['source_dir']}",
            f"Tags:        {tags}",
        ])


def build_plan(cfg: LaunchConfig) -> LaunchPlan:
    job = cfg.job
    command = [str(part).format(**cfg.variables) for part in job.command]
    hyperparameters = {
        COMMAND_KEY: json.dumps(command),
        OUTPUTS_KEY: json.dumps(job.output_dirs),
    }
    kwargs: dict[str, Any] = {
        "entry_point": ENTRY_FILENAME,
        "source_dir": job.source_dir,
        "role": cfg.aws.role_arn,
        "instance_type": job.instance_type,
        "instance_count": job.instance_count,
        "framework": job.framework,
        "framework_version": job.framework_version,
        "py_version": job.py_version,
        "max_run": int(job.max_run_hours * 3600),
        "environment": dict(job.environment),
    }
    if job.use_spot:
        kwargs["use_spot_instances"] = True
        kwargs["max_wait"] = int((job.max_wait_hours or job.max_run_hours) * 3600)
    return LaunchPlan(
        base_job_name=build_job_name(job, cfg.variables),
        command=command,
        hyperparameters=hyperparameters,
        tags=build_tags(job, cfg.variables),
        estimator_kwargs=kwargs,
    )


def run_preflight(cfg: LaunchConfig) -> int:
    """Executa job.preflight no source_dir (ex.: validar configs) antes de gastar com a AWS."""
    if not cfg.job.preflight:
        return 0
    import subprocess
    import sys
    cmd = [str(part).format(**cfg.variables) for part in cfg.job.preflight]
    if cmd[0] in ("python", "python3"):
        cmd = [sys.executable, *cmd[1:]]
    return subprocess.run(cmd, cwd=cfg.job.source_dir).returncode


@contextmanager
def staged_entry(source_dir: str) -> Iterator[Path]:
    """Coloca o entry point no source_dir durante o upload e remove depois, se fomos nós que criamos."""
    target = Path(source_dir) / ENTRY_FILENAME
    created = not target.exists()
    shutil.copy2(Path(__file__).with_name("entry.py"), target)
    try:
        yield target
    finally:
        if created:
            target.unlink(missing_ok=True)


_FRAMEWORKS = {
    "pytorch": ("sagemaker.pytorch", "PyTorch"),
    "tensorflow": ("sagemaker.tensorflow", "TensorFlow"),
    "huggingface": ("sagemaker.huggingface", "HuggingFace"),
}


def make_estimator(plan: LaunchPlan, sagemaker_session):
    import importlib
    kwargs = dict(plan.estimator_kwargs)
    framework = kwargs.pop("framework")
    if framework not in _FRAMEWORKS:
        raise ValueError(f"framework não suportado: {framework} (use {sorted(_FRAMEWORKS)})")
    module, cls = _FRAMEWORKS[framework]
    estimator_cls = getattr(importlib.import_module(module), cls)
    return estimator_cls(base_job_name=plan.base_job_name, hyperparameters=plan.hyperparameters,
                         tags=plan.tags, sagemaker_session=sagemaker_session, **kwargs)


def launch(cfg: LaunchConfig, boto_session, wait: bool = True,
           estimator_factory: Callable = make_estimator) -> str:
    """Lança o job e devolve o nome definitivo (com o sufixo de timestamp da AWS)."""
    import sagemaker  # import tardio: só quem lança precisa do SDK

    plan = build_plan(cfg)
    session = sagemaker.Session(boto_session=boto_session)
    estimator = estimator_factory(plan, session)
    with staged_entry(cfg.job.source_dir):
        estimator.fit(wait=wait, logs="All" if wait else "None")
    name = estimator.latest_training_job.name
    Path(LAST_JOB_FILE).write_text(name, encoding="utf-8")
    return name


@dataclass
class FetchResult:
    job_dir: Path
    log_file: Path | None
    collected: dict[str, list[Path]]
    mlflow_experiments: int


def fetch(cfg: LaunchConfig, boto_session, job_name: str, desc: dict | None = None,
          include_logs: bool = True) -> FetchResult:
    """Baixa logs e artefatos de um job terminado e organiza em outputs.dir/<job>/."""
    sm = boto_session.client("sagemaker")
    desc = desc or jobs.describe(sm, job_name)
    job_dir = Path(cfg.outputs.dir) / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    if include_logs:
        events = logs.fetch_events(boto_session.client("logs"), job_name)
        if events:
            log_file = logs.write_log(events, job_dir / "training.log")

    collected: dict[str, list[Path]] = {}
    n_exp = 0
    uri = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if uri:
        tar = artifacts.download_s3_object(boto_session.client("s3"), uri, job_dir / "model.tar.gz")
        extracted = job_dir / "extracted"
        if extracted.exists():
            shutil.rmtree(extracted)
        artifacts.safe_extract(tar, extracted)
        collected = artifacts.collect(extracted, job_dir, cfg.outputs.collect)
        if cfg.outputs.mlflow_import:
            n_exp = artifacts.merge_mlruns(extracted / "mlruns", Path(cfg.outputs.mlflow_local_dir))
    return FetchResult(job_dir=job_dir, log_file=log_file, collected=collected, mlflow_experiments=n_exp)


def last_job_name() -> str | None:
    p = Path(LAST_JOB_FILE)
    return p.read_text(encoding="utf-8").strip() or None if p.exists() else None
