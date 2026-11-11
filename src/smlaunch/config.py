"""Leitura e validação do YAML de lançamento.

Um arquivo de lançamento descreve *onde* e *como* o treino roda; o que o treino
faz continua sendo responsabilidade do projeto (o `command`). Valores no formato
``${VAR}`` ou ``${VAR:-padrão}`` são expandidos a partir do ambiente, para que
nenhuma credencial ou ARN precise ficar no arquivo.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Limite do SageMaker para base_job_name: a AWS acrescenta um sufixo de timestamp
# (~24 caracteres) e o nome final não pode passar de 63.
MAX_BASE_JOB_NAME = 39


class ConfigError(ValueError):
    """Configuração de lançamento inválida."""


def deep_merge(base: dict, override: dict) -> dict:
    """Mescla dicionários recursivamente; `override` vence em conflitos."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Expande ``${VAR}``/``${VAR:-padrão}`` em strings, listas e dicionários."""
    env = os.environ if env is None else env
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in env:
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"variável de ambiente '{name}' não definida")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    return value


@dataclass
class AwsConfig:
    region: str = "us-east-1"
    role_arn: str = ""
    profile: str | None = None  # None = cadeia padrão de credenciais do boto3


@dataclass
class JobConfig:
    name_template: str = "{project}-{experiment}"
    name_hook: str | None = None  # "modulo:funcao" que recebe (variables) e devolve o nome base
    instance_type: str = "ml.g4dn.xlarge"
    instance_count: int = 1
    max_run_hours: float = 24.0
    use_spot: bool = False
    max_wait_hours: float | None = None
    framework: str = "pytorch"
    framework_version: str = "2.1"
    py_version: str = "py310"
    source_dir: str = "."
    command: list[str] = field(default_factory=list)
    preflight: list[str] = field(default_factory=list)  # roda localmente antes do envio; != 0 aborta
    environment: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    output_dirs: list[str] = field(default_factory=lambda: ["runs"])


@dataclass
class OutputConfig:
    dir: str = "./output"
    collect: dict[str, str] = field(default_factory=lambda: {"checkpoints": "**/*.pt"})
    mlflow_import: bool = True
    mlflow_local_dir: str = "./mlruns"


@dataclass
class LaunchConfig:
    aws: AwsConfig
    job: JobConfig
    outputs: OutputConfig
    variables: dict[str, Any] = field(default_factory=dict)
    pricing: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        errors: list[str] = []
        if not self.aws.role_arn:
            errors.append("aws.role_arn vazio (use ${SAGEMAKER_ROLE_ARN})")
        elif not re.match(r"^arn:aws[a-z-]*:iam::\d{12}:role/.+", self.aws.role_arn):
            errors.append(f"aws.role_arn com formato inválido: {self.aws.role_arn!r}")
        if not self.job.command:
            errors.append("job.command vazio: informe o comando de treino")
        if self.job.instance_count < 1:
            errors.append("job.instance_count deve ser >= 1")
        if self.job.max_run_hours <= 0:
            errors.append("job.max_run_hours deve ser > 0")
        if self.job.use_spot:
            wait = self.job.max_wait_hours
            if wait is None or wait < self.job.max_run_hours:
                errors.append("com use_spot, job.max_wait_hours deve ser >= max_run_hours")
        if not Path(self.job.source_dir).is_dir():
            errors.append(f"job.source_dir não existe: {self.job.source_dir}")
        if errors:
            raise ConfigError("; ".join(errors))


def _build(section: dict | None, cls):
    section = section or {}
    known = set(cls.__dataclass_fields__)
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"chaves desconhecidas em {cls.__name__}: {sorted(unknown)}")
    return cls(**section)


def load_config(
    path: str | Path,
    overrides: dict | None = None,
    env: dict[str, str] | None = None,
) -> LaunchConfig:
    """Carrega um YAML de lançamento (com ``extends:`` opcional) e expande variáveis."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        base_raw = yaml.safe_load((path.parent / parent).read_text(encoding="utf-8")) or {}
        base_raw.pop("extends", None)
        raw = deep_merge(base_raw, raw)
    if overrides:
        raw = deep_merge(raw, overrides)
    raw = expand_env(raw, env)

    unknown = set(raw) - {"aws", "job", "outputs", "variables", "pricing"}
    if unknown:
        raise ConfigError(f"seções desconhecidas: {sorted(unknown)}")

    job_raw = dict(raw.get("job") or {})
    # source_dir relativo ao arquivo de configuração
    src = job_raw.get("source_dir", ".")
    if not Path(src).is_absolute():
        job_raw["source_dir"] = str((path.parent / src).resolve())

    return LaunchConfig(
        aws=_build(raw.get("aws"), AwsConfig),
        job=_build(job_raw, JobConfig),
        outputs=_build(raw.get("outputs"), OutputConfig),
        variables=dict(raw.get("variables") or {}),
        pricing={k: float(v) for k, v in (raw.get("pricing") or {}).items()},
    )
