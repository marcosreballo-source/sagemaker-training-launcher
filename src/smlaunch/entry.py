"""Entry point executado DENTRO do container do SageMaker.

Só usa a biblioteca padrão: o launcher copia este arquivo para o source_dir como
``_smlaunch_entry.py``, então o container não precisa ter o smlaunch instalado.

Fluxo: lê o comando de treino dos hiperparâmetros, aponta o MLflow para um file
store local, executa o comando e copia as saídas (e o mlruns) para SM_MODEL_DIR,
que o SageMaker empacota em model.tar.gz no S3. O código de saída do treino é
propagado, então o job aparece como Failed quando o treino falha.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

COMMAND_KEY = "smlaunch_command"
OUTPUTS_KEY = "smlaunch_output_dirs"


def _decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def read_hyperparameters(environ=os.environ) -> dict:
    hps = json.loads(environ.get("SM_HPS", "{}") or "{}")
    return {k: _decode(v) for k, v in hps.items()}


def copy_outputs(output_dirs: list[str], workdir: Path, model_dir: Path) -> list[Path]:
    copied = []
    for entry in output_dirs:
        src = Path(entry) if Path(entry).is_absolute() else workdir / entry
        if src.exists():
            dst = model_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def main(environ=os.environ) -> int:
    workdir = Path(__file__).resolve().parent
    model_dir = Path(environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    hps = read_hyperparameters(environ)

    command = hps.get(COMMAND_KEY)
    if not isinstance(command, list) or not command:
        print(f"[smlaunch] hiperparâmetro '{COMMAND_KEY}' ausente ou inválido", flush=True)
        return 2
    output_dirs = hps.get(OUTPUTS_KEY) or []

    env = dict(environ)
    mlruns = workdir / "mlruns"
    env.setdefault("MLFLOW_TRACKING_URI", mlruns.as_uri())
    if command[0] in ("python", "python3"):
        command = [sys.executable, *command[1:]]

    print(f"[smlaunch] cwd: {workdir}", flush=True)
    print(f"[smlaunch] comando: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=workdir, env=env)

    model_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_outputs([*output_dirs, str(mlruns)], workdir, model_dir)
    print(f"[smlaunch] saídas copiadas para {model_dir}: {[p.name for p in copied]}", flush=True)
    print(f"[smlaunch] código de saída do treino: {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
