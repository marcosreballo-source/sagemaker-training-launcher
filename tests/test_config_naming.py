import pytest

from smlaunch.config import ConfigError, deep_merge, expand_env, load_config
from smlaunch.naming import build_job_name, build_tags, sanitize_job_name

from conftest import ROLE


def test_deep_merge_overrides_nested_keys():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_expand_env_uses_default_and_raises_when_missing():
    assert expand_env("${X:-padrao}", env={}) == "padrao"
    assert expand_env(["${X}"], env={"X": "1"}) == ["1"]
    with pytest.raises(ConfigError):
        expand_env("${NAO_EXISTE}", env={})


def test_load_config_resolves_source_dir_and_role(project):
    cfg = load_config(project / "smlaunch.yaml", env={"SAGEMAKER_ROLE_ARN": ROLE})
    assert cfg.aws.role_arn == ROLE
    assert cfg.job.source_dir == str((project / "train").resolve())
    cfg.validate()


def test_validate_rejects_missing_role_and_bad_spot(project):
    cfg = load_config(project / "smlaunch.yaml", env={},
                      overrides={"job": {"use_spot": True, "max_wait_hours": 1}})
    with pytest.raises(ConfigError) as exc:
        cfg.validate()
    assert "role_arn" in str(exc.value) and "max_wait_hours" in str(exc.value)


def test_unknown_keys_are_rejected(project):
    with pytest.raises(ConfigError):
        load_config(project / "smlaunch.yaml", env={}, overrides={"job": {"instnce_type": "x"}})


def test_extends_merges_parent(project):
    (project / "exp.yaml").write_text("extends: smlaunch.yaml\nvariables:\n  experiment: ablacao\n")
    cfg = load_config(project / "exp.yaml", env={})
    assert cfg.variables["experiment"] == "ablacao"
    assert cfg.variables["project"] == "demo"


def test_job_name_template_sanitized_and_truncated(project):
    cfg = load_config(project / "smlaunch.yaml", env={})
    assert build_job_name(cfg.job, cfg.variables) == "demo-baseline"
    assert sanitize_job_name("--a_b  c--") == "a-b-c"
    with pytest.warns(UserWarning):
        assert len(sanitize_job_name("x" * 60)) == 39


def test_job_name_missing_variable(project):
    cfg = load_config(project / "smlaunch.yaml", env={})
    cfg.job.name_template = "{project}-{nao_tem}"
    with pytest.raises(ConfigError):
        build_job_name(cfg.job, cfg.variables)


def test_job_name_hook(project, monkeypatch, tmp_path):
    (tmp_path / "meu_hook.py").write_text("def nome(v):\n    return v['project'] + '-hook'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = load_config(project / "smlaunch.yaml", env={})
    cfg.job.name_hook = "meu_hook:nome"
    assert build_job_name(cfg.job, cfg.variables) == "demo-hook"


def test_tags_aws_format(project):
    cfg = load_config(project / "smlaunch.yaml", env={})
    assert {"Key": "Project", "Value": "demo"} in build_tags(cfg.job)
    cfg.job.tags = {"Exp": "{experiment}"}
    assert build_tags(cfg.job, cfg.variables) == [{"Key": "Exp", "Value": "baseline"}]


def test_example_config_is_valid(monkeypatch):
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / "examples" / "smlaunch.yaml"
    cfg = load_config(example, env={"SAGEMAKER_ROLE_ARN": ROLE})
    cfg.validate()
    assert build_job_name(cfg.job, cfg.variables) == "demo-research-baseline"
