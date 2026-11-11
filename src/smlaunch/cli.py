"""CLI: smlaunch launch | status | wait | fetch | list | audit | cost."""

from __future__ import annotations

import argparse
import sys

from smlaunch import audit, jobs, launcher
from smlaunch.aws import make_session
from smlaunch.config import ConfigError, LaunchConfig, load_config


def _parse_vars(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ConfigError(f"--var espera chave=valor, recebido {pair!r}")
        out[key] = value
    return out


def _load(args) -> LaunchConfig:
    overrides: dict = {}
    if getattr(args, "var", None):
        overrides["variables"] = _parse_vars(args.var)
    if getattr(args, "instance", None):
        overrides.setdefault("job", {})["instance_type"] = args.instance
    if getattr(args, "spot", False):
        overrides.setdefault("job", {})["use_spot"] = True
    return load_config(args.config, overrides=overrides)


def _job_name(args) -> str:
    name = getattr(args, "job", None) or launcher.last_job_name()
    if not name:
        raise SystemExit("informe o nome do job (ou lance um antes para usar o último)")
    return name


def cmd_launch(args) -> int:
    cfg = _load(args)
    cfg.validate()
    plan = launcher.build_plan(cfg)
    print(plan.render())
    rc = launcher.run_preflight(cfg)
    if rc != 0:
        print(f"\nPreflight falhou (código {rc}): job NÃO enviado.", file=sys.stderr)
        return rc
    if args.dry_run:
        print("\n--dry-run: nada foi enviado à AWS.")
        return 0
    session = make_session(cfg.aws)
    name = launcher.launch(cfg, session, wait=not args.no_wait)
    print(f"\nJob: {name}")
    if args.no_wait:
        print("Rodando na AWS. Depois: smlaunch wait / smlaunch fetch")
        return 0
    return _finish(cfg, session, name, fetch=not args.no_fetch)


def _finish(cfg: LaunchConfig, session, name: str, fetch: bool) -> int:
    desc = jobs.describe(session.client("sagemaker"), name)
    summary = jobs.summarize(desc, cfg.pricing)
    print(summary.render())
    if fetch and summary.status in jobs.TERMINAL_STATUSES:
        res = launcher.fetch(cfg, session, name, desc)
        print(f"\nSaída em: {res.job_dir}")
        for dest, files in res.collected.items():
            print(f"  {dest}: {len(files)} arquivo(s)")
        if res.log_file:
            print(f"  log: {res.log_file}")
        if res.mlflow_experiments:
            print(f"  MLflow: {res.mlflow_experiments} experimento(s) em {cfg.outputs.mlflow_local_dir}"
                  " (mlflow ui para visualizar)")
    return 0 if summary.status == "Completed" else 1


def cmd_status(args) -> int:
    cfg = _load(args)
    session = make_session(cfg.aws)
    print(jobs.summarize(jobs.describe(session.client("sagemaker"), _job_name(args)), cfg.pricing).render())
    return 0


def cmd_wait(args) -> int:
    cfg = _load(args)
    session = make_session(cfg.aws)
    name = _job_name(args)
    jobs.wait(session.client("sagemaker"), name,
              on_change=lambda s, sub: print(f"  {s} / {sub}", flush=True))
    return _finish(cfg, session, name, fetch=not args.no_fetch)


def cmd_fetch(args) -> int:
    cfg = _load(args)
    return _finish(cfg, make_session(cfg.aws), _job_name(args), fetch=True)


def cmd_list(args) -> int:
    cfg = _load(args)
    rows = jobs.list_recent(make_session(cfg.aws).client("sagemaker"), args.limit, args.contains)
    for j in rows:
        print(f"{j['TrainingJobName']:<64} {j['TrainingJobStatus']:<11} {j['CreationTime']:%Y-%m-%d %H:%M}")
    return 0


def cmd_audit(args) -> int:
    cfg = _load(args)
    res = audit.find_active_resources(make_session(cfg.aws))
    print(res.render())
    return 0 if res.all_clear else 3


def cmd_cost(args) -> int:
    cfg = _load(args)
    print(audit.month_cost(make_session(cfg.aws), tag_key=args.tag).render())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smlaunch", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("-c", "--config", default="smlaunch.yaml", help="YAML de lançamento")
        sp.add_argument("--var", action="append", help="sobrescreve variables (chave=valor)")
        sp.set_defaults(func=fn)
        return sp

    sp = add("launch", cmd_launch, "lança um Training Job")
    sp.add_argument("--dry-run", action="store_true", help="mostra o plano sem chamar a AWS")
    sp.add_argument("--no-wait", action="store_true", help="lança e sai (o job segue na nuvem)")
    sp.add_argument("--no-fetch", action="store_true", help="não baixa artefatos ao terminar")
    sp.add_argument("--instance", help="sobrescreve job.instance_type")
    sp.add_argument("--spot", action="store_true", help="usa instâncias spot")

    for name, fn, help_ in [("status", cmd_status, "mostra status e custo de um job"),
                            ("wait", cmd_wait, "aguarda o job e baixa os resultados"),
                            ("fetch", cmd_fetch, "baixa logs, artefatos e runs do MLflow")]:
        sp = add(name, fn, help_)
        sp.add_argument("job", nargs="?", help="nome do job (padrão: o último lançado)")
        if name == "wait":
            sp.add_argument("--no-fetch", action="store_true")

    sp = add("list", cmd_list, "lista os jobs recentes")
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--contains", help="filtra por trecho do nome")

    add("audit", cmd_audit, "procura recursos ainda ativos que geram custo")
    sp = add("cost", cmd_cost, "gasto do mês no Cost Explorer")
    sp.add_argument("--tag", help="agrupa também por esta tag de custo (ex.: Project)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"Configuração inválida: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
