import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smlaunch import jobs, launcher
from smlaunch.config import load_config
from smlaunch.entry import COMMAND_KEY, OUTPUTS_KEY

from conftest import ROLE


@pytest.fixture
def cfg(project, monkeypatch):
    monkeypatch.chdir(project)
    return load_config(project / "smlaunch.yaml", env={"SAGEMAKER_ROLE_ARN": ROLE})


def test_plan_formats_command_and_hyperparameters(cfg):
    plan = launcher.build_plan(cfg)
    assert plan.base_job_name == "demo-baseline"
    assert plan.command == ["python", "main.py", "--config", "configs/base.yaml"]
    assert json.loads(plan.hyperparameters[COMMAND_KEY]) == plan.command
    assert json.loads(plan.hyperparameters[OUTPUTS_KEY]) == ["runs"]
    assert plan.estimator_kwargs["max_run"] == 7200
    assert "use_spot_instances" not in plan.estimator_kwargs
    assert "demo-baseline" in plan.render()


def test_plan_spot(cfg):
    cfg.job.use_spot, cfg.job.max_wait_hours = True, 3
    kw = launcher.build_plan(cfg).estimator_kwargs
    assert kw["use_spot_instances"] is True and kw["max_wait"] == 10800


def test_staged_entry_is_removed_after_upload(cfg):
    target = Path(cfg.job.source_dir) / launcher.ENTRY_FILENAME
    with launcher.staged_entry(cfg.job.source_dir) as p:
        assert p.exists() and "smlaunch_command" in p.read_text(encoding="utf-8")
    assert not target.exists()


def test_launch_uses_estimator_and_records_last_job(cfg, monkeypatch):
    fake_sagemaker = types.ModuleType("sagemaker")
    fake_sagemaker.Session = lambda boto_session: "sm-session"
    monkeypatch.setitem(sys.modules, "sagemaker", fake_sagemaker)

    seen = {}

    class FakeEstimator:
        latest_training_job = types.SimpleNamespace(name="demo-baseline-2026-11-11-12-00-00-000")

        def fit(self, wait, logs):
            entry = Path(cfg.job.source_dir) / launcher.ENTRY_FILENAME
            seen.update(wait=wait, logs=logs, entry_present=entry.exists())

    def factory(plan, session):
        seen["session"] = session
        return FakeEstimator()

    name = launcher.launch(cfg, boto_session=object(), wait=False, estimator_factory=factory)
    assert name.startswith("demo-baseline")
    assert seen == {"session": "sm-session", "wait": False, "logs": "None", "entry_present": True}
    assert launcher.last_job_name() == name


def test_summarize_cost_and_duration():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    desc = {
        "TrainingJobName": "j", "TrainingJobStatus": "Completed", "SecondaryStatus": "Completed",
        "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1},
        "TrainingStartTime": t0, "TrainingEndTime": t0 + timedelta(hours=2),
        "BillableTimeInSeconds": 7200,
    }
    s = jobs.summarize(desc, pricing={"ml.g4dn.xlarge": 1.0})
    assert s.duration_s == 7200 and s.estimated_cost_usd == pytest.approx(2.0)
    assert jobs.estimate_cost("ml.instancia-desconhecida", 100) is None


def test_wait_backs_off_until_terminal():
    states = iter(["InProgress", "InProgress", "InProgress", "Completed"])

    class SM:
        def describe_training_job(self, TrainingJobName):
            return {"TrainingJobName": TrainingJobName, "TrainingJobStatus": next(states),
                    "SecondaryStatus": "x"}

    sleeps, changes = [], []
    desc = jobs.wait(SM(), "j", poll_s=10, max_poll_s=20, sleep=sleeps.append,
                     on_change=lambda *s: changes.append(s))
    assert desc["TrainingJobStatus"] == "Completed"
    assert sleeps == [10, 15, 20]
    assert changes == [("InProgress", "x"), ("Completed", "x")]


def test_describe_not_found():
    class SM:
        def describe_training_job(self, TrainingJobName):
            raise RuntimeError("An error occurred (ValidationException): Requested resource not found")

    with pytest.raises(jobs.JobNotFoundError):
        jobs.describe(SM(), "x")
