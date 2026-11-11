from smlaunch.cli import main

from conftest import ROLE


def test_dry_run_prints_plan_without_aws(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", ROLE)
    rc = main(["launch", "--dry-run", "--var", "experiment=ablacao", "--instance", "ml.g5.xlarge"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "demo-ablacao" in out and "ml.g5.xlarge" in out and "nada foi enviado" in out


def test_preflight_failure_blocks_launch(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    monkeypatch.setenv("SAGEMAKER_ROLE_ARN", ROLE)
    (project / "train" / "check.py").write_text("import sys; sys.exit(sys.argv[1] != 'configs/base.yaml')\n")
    cfg = (project / "smlaunch.yaml").read_text()
    (project / "smlaunch.yaml").write_text(cfg.replace(
        "  output_dirs: [runs]", '  output_dirs: [runs]\n  preflight: ["python", "check.py", "{config}"]'))
    assert main(["launch", "--dry-run"]) == 0
    assert main(["launch", "--dry-run", "--var", "config=errado.yaml"]) == 1
    assert "NÃO enviado" in capsys.readouterr().err


def test_invalid_config_returns_2(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    monkeypatch.delenv("SAGEMAKER_ROLE_ARN", raising=False)
    assert main(["launch", "--dry-run"]) == 2
    assert "role_arn" in capsys.readouterr().err
