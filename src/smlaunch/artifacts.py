"""Download e organização dos artefatos de um job (model.tar.gz no S3)."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path, PurePosixPath

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class UnsafeArchiveError(RuntimeError):
    pass


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"URI S3 inválida: {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ValueError(f"URI S3 inválida: {uri!r}")
    return bucket, key


def _http_session() -> requests.Session:
    # Retry em 307 cobre o redirecionamento regional do S3 logo após o upload.
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[307, 429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_s3_object(s3_client, uri: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Baixa via URL pré-assinada, em streaming, com retry."""
    bucket, key = parse_s3_uri(uri)
    url = s3_client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                           ExpiresIn=3600)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _http_session().get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
    tmp.replace(dest)  # só aparece completo
    return dest


def safe_extract(archive: Path, target: Path) -> list[Path]:
    """Extrai um tar.gz recusando caminhos absolutos, '..' e links (path traversal)."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for m in members:
            p = PurePosixPath(m.name)
            if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk() or m.isdev():
                raise UnsafeArchiveError(f"entrada insegura no arquivo: {m.name!r}")
            dest = (target / m.name).resolve()
            if target not in dest.parents and dest != target:
                raise UnsafeArchiveError(f"entrada fora do destino: {m.name!r}")
        # Defesa em profundidade: o filtro "data" do tarfile (Python >= 3.12 e backports)
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        for m in members:
            tar.extract(m, target, **extra)
            if m.isfile():
                extracted.append(target / m.name)
    return extracted


def collect(extracted_root: Path, job_dir: Path, patterns: dict[str, str]) -> dict[str, list[Path]]:
    """Copia arquivos que casam com cada glob para `job_dir/<destino>` (achatando a árvore)."""
    result: dict[str, list[Path]] = {}
    for dest_name, pattern in patterns.items():
        dest = job_dir if dest_name in (".", "") else job_dir / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for src in sorted(extracted_root.glob(pattern)):
            if src.is_file():
                out = dest / src.name
                shutil.copy2(src, out)
                copied.append(out)
        result[dest_name] = copied
    return result


def merge_mlruns(remote: Path, local: Path) -> int:
    """Mescla um file store do MLflow vindo do job no file store local. Retorna nº de experimentos."""
    if not remote.is_dir():
        return 0
    local.mkdir(parents=True, exist_ok=True)
    for item in remote.iterdir():
        dst = local / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    return sum(1 for d in local.iterdir() if d.is_dir() and d.name not in {".trash", "models"})
