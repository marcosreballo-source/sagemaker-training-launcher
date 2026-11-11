import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from smlaunch import artifacts, entry, logs


def _make_tar(path: Path, files: dict[str, bytes], symlink: str | None = None) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
    return path


def test_parse_s3_uri():
    assert artifacts.parse_s3_uri("s3://b/k/model.tar.gz") == ("b", "k/model.tar.gz")
    with pytest.raises(ValueError):
        artifacts.parse_s3_uri("https://b/k")


def test_safe_extract_collect_and_mlflow(tmp_path):
    tar = _make_tar(tmp_path / "model.tar.gz", {
        "runs/exp1/final.pt": b"w",
        "runs/exp1/config.resolved.yaml": b"a: 1",
        "mlruns/123/meta.yaml": b"name: exp",
    })
    extracted = tmp_path / "x"
    files = artifacts.safe_extract(tar, extracted)
    assert len(files) == 3

    job_dir = tmp_path / "job"
    out = artifacts.collect(extracted, job_dir, {"checkpoints": "**/*.pt", ".": "**/config.resolved.yaml"})
    assert (job_dir / "checkpoints" / "final.pt").exists()
    assert (job_dir / "config.resolved.yaml").exists()
    assert len(out["checkpoints"]) == 1

    local = tmp_path / "mlruns"
    assert artifacts.merge_mlruns(extracted / "mlruns", local) == 1
    assert artifacts.merge_mlruns(tmp_path / "nao-existe", local) == 0


@pytest.mark.parametrize("bad", ["../escape.txt", "/abs.txt"])
def test_safe_extract_rejects_traversal(tmp_path, bad):
    tar = _make_tar(tmp_path / "bad.tar.gz", {bad: b"x"})
    with pytest.raises(artifacts.UnsafeArchiveError):
        artifacts.safe_extract(tar, tmp_path / "x")


def test_safe_extract_rejects_symlink(tmp_path):
    tar = _make_tar(tmp_path / "bad.tar.gz", {"ok.txt": b"x"}, symlink="link")
    with pytest.raises(artifacts.UnsafeArchiveError):
        artifacts.safe_extract(tar, tmp_path / "x")
    assert not (tmp_path / "x" / "ok.txt").exists()  # nada é extraído se algo é inseguro


def test_fetch_log_events_paginates_and_sorts(tmp_path):
    class Logs:
        def describe_log_streams(self, **kw):
            if "nextToken" not in kw:
                return {"logStreams": [{"logStreamName": "j/algo-1"}], "nextToken": "t"}
            return {"logStreams": [{"logStreamName": "j/algo-2"}]}

        def get_log_events(self, **kw):
            pages = {
                ("j/algo-1", None): {"events": [{"timestamp": 2000, "message": "b"}], "nextForwardToken": "f1"},
                ("j/algo-1", "f1"): {"events": [], "nextForwardToken": "f1"},
                ("j/algo-2", None): {"events": [{"timestamp": 1000, "message": "a"}], "nextForwardToken": "g1"},
                ("j/algo-2", "g1"): {"events": [], "nextForwardToken": "g1"},
            }
            return pages[(kw["logStreamName"], kw.get("nextToken"))]

    events = logs.fetch_events(Logs(), "j")
    assert [e["message"] for e in events] == ["a", "b"]
    text = logs.write_log(events, tmp_path / "t.log").read_text(encoding="utf-8")
    assert text.splitlines()[0].endswith("] a")


def test_entry_runs_command_and_copies_outputs(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    entry_file = code / "_smlaunch_entry.py"
    entry_file.write_text(Path(entry.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (code / "train.py").write_text(
        "import os, pathlib\n"
        "p = pathlib.Path('runs'); p.mkdir(); (p / 'm.pt').write_text('w')\n"
        "pathlib.Path('mlruns').mkdir(); pathlib.Path('mlruns/x').write_text(os.environ['MLFLOW_TRACKING_URI'])\n"
        "raise SystemExit(3)\n"
    )
    model_dir = tmp_path / "model"
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
    env.update({
        "SM_MODEL_DIR": str(model_dir),
        "SM_HPS": json.dumps({entry.COMMAND_KEY: json.dumps(["python", "train.py"]),
                              entry.OUTPUTS_KEY: json.dumps(["runs"])}),
    })
    monkeypatch.syspath_prepend(str(code))
    import importlib
    staged = importlib.import_module("_smlaunch_entry")
    rc = staged.main(environ=env)
    assert rc == 3  # o código de saída do treino é propagado
    assert (model_dir / "runs" / "m.pt").exists()
    assert (model_dir / "mlruns" / "x").read_text().startswith("file:")


def test_entry_without_command_fails():
    assert entry.main(environ={"SM_HPS": "{}", "SM_MODEL_DIR": "/tmp/nao-usado"}) == 2
