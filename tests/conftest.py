from pathlib import Path

import pytest

ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Projeto mínimo com source_dir e um YAML de lançamento."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "main.py").write_text("print('ok')\n")
    (tmp_path / "smlaunch.yaml").write_text(
        """
aws:
  region: us-east-2
  role_arn: ${SAGEMAKER_ROLE_ARN:-}
job:
  name_template: "{project}-{experiment}"
  instance_type: ml.g4dn.xlarge
  max_run_hours: 2
  source_dir: train
  command: ["python", "main.py", "--config", "{config}"]
  tags: {Project: demo, Squad: research}
  output_dirs: [runs]
variables:
  project: demo
  experiment: baseline
  config: configs/base.yaml
outputs:
  dir: out
  collect: {checkpoints: "**/*.pt", ".": "**/config.resolved.yaml"}
pricing:
  ml.g4dn.xlarge: 1.0
""",
        encoding="utf-8",
    )
    return tmp_path
