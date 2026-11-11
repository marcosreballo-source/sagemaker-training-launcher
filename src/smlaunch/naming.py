"""Nome do job e tags de custo.

O nome é gerado a partir de um template com as variáveis do YAML
(``{project}-{experiment}``) ou, quando o projeto precisa de algo mais rico,
por um hook ``modulo:funcao`` que recebe as variáveis e devolve o nome.
"""

from __future__ import annotations

import importlib
import re
import string
import sys
import warnings
from typing import Any, Callable

from smlaunch.config import MAX_BASE_JOB_NAME, ConfigError, JobConfig

_INVALID = re.compile(r"[^A-Za-z0-9-]+")


def sanitize_job_name(name: str, limit: int = MAX_BASE_JOB_NAME) -> str:
    """Ajusta o nome às regras do SageMaker: [A-Za-z0-9-], sem hífen nas pontas, até `limit`."""
    cleaned = _INVALID.sub("-", name).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    if not cleaned:
        raise ConfigError("nome de job vazio após sanitização")
    if len(cleaned) > limit:
        warnings.warn(f"nome do job truncado para {limit} caracteres: {cleaned!r}", stacklevel=2)
        cleaned = cleaned[:limit].rstrip("-")
    return cleaned


def _load_hook(spec: str, search_dir: str | None = None) -> Callable[[dict[str, Any]], str]:
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise ConfigError(f"name_hook deve ter o formato 'modulo:funcao', recebido {spec!r}")
    # O hook costuma morar no próprio projeto (source_dir), que não está no sys.path da CLI.
    if search_dir and search_dir not in sys.path:
        sys.path.insert(0, search_dir)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def build_job_name(job: JobConfig, variables: dict[str, Any]) -> str:
    if job.name_hook:
        raw = _load_hook(job.name_hook, job.source_dir)(variables)
    else:
        fields = {f for _, f, _, _ in string.Formatter().parse(job.name_template) if f}
        missing = fields - set(variables)
        if missing:
            raise ConfigError(f"variáveis ausentes para name_template: {sorted(missing)}")
        raw = job.name_template.format(**variables)
    return sanitize_job_name(str(raw))


def build_tags(job: JobConfig, variables: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Tags no formato da API da AWS (valores aceitam {variaveis}), para rastrear custo por projeto/time."""
    variables = variables or {}
    return [{"Key": str(k), "Value": str(v).format(**variables)} for k, v in job.tags.items()]
