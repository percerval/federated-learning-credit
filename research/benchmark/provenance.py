"""Metadados e identificadores determinísticos para futuras execuções.

A identidade científica de uma execução é distinguida dos metadados
operacionais mutáveis:

- ``scientific_identity``: definição imutável do experimento (protocolo, spec,
  dados, configuração, training_seed, commit Git e hashes SHA-256 dos
  arquivos-fonte executados). É ela que forma o ``run_id``. Dois estados dirty
  distintos sobre o mesmo commit diferem pelos hashes dos arquivos; a criação
  de artefatos de saída — que apenas muda o estado dirty — não a altera.
- ``operational``: estado Git dirty (com linhas do porcelain), ambiente e
  relatório de determinismo, registrados para vínculo probatório sem participar
  do hash de identidade.

Todos os caminhos registrados são relativos à raiz do repositório e validados
por ``contracts.validation.relative_path``.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

from contracts import ContractError, validate_protocol
from contracts.validation import json_value, relative_path, strict_json
from research.benchmark.seeding import TRAINING_SEEDS, canonical_json_bytes
from research.data.pipeline import load_record

# Manifesto default completo dos arquivos .py executados atualmente pela
# infraestrutura E3.1A, pelos modelos clássicos E3.1B.1, pelos componentes
# MLP E3.1B.2 e pela agregação FedAvg E3.1B.3: pacote
# research.benchmark, suas dependências em research.data e contracts, e o
# pacote pai research. Um teste de cobertura falha se um novo .py aparecer
# sob esses diretórios sem ser registrado aqui. A E3.2 deverá acrescentar ao
# manifesto (via source_files) os novos arquivos executados — coordenador,
# runner e demais módulos — antes de qualquer execução oficial.
DEFAULT_SOURCE_FILES = (
    "research/__init__.py",
    "research/benchmark/__init__.py",
    "research/benchmark/classic.py",
    "research/benchmark/client_test.py",
    "research/benchmark/data.py",
    "research/benchmark/evaluation.py",
    "research/benchmark/fedavg.py",
    "research/benchmark/mlp.py",
    "research/benchmark/provenance.py",
    "research/benchmark/seeding.py",
    "research/data/__init__.py",
    "research/data/partitioning.py",
    "research/data/pipeline.py",
    "research/data/preprocessing.py",
    "research/data/splitting.py",
    "contracts/__init__.py",
    "contracts/schemas.py",
    "contracts/validation.py",
)


class ProvenanceError(ValueError):
    """Proveniência incompleta ou incompatível com a especificação congelada."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=repo_root, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"falha ao consultar Git: {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _inside_root(root: Path, path: Path) -> Path:
    """Resolve ``path`` e exige que exista como arquivo dentro de ``root``."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"caminho fora da raiz do repositório: {path}") from exc
    _require(resolved.is_file(), f"arquivo inexistente ou não regular: {path}")
    return root / relative


def _safe_relative(root: Path, path: Path) -> str:
    """Caminho relativo POSIX dentro da raiz, compatível com os contratos."""
    resolved = _inside_root(root, path)
    value = resolved.relative_to(root).as_posix()
    try:
        relative_path(value, "source_file.path")
    except ContractError as exc:
        raise ProvenanceError(str(exc)) from exc
    return value


def environment_snapshot(determinism_report: Mapping | None = None) -> dict:
    packages = {}
    limitations = []
    for distribution in ("torch", "numpy", "scikit-learn"):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
            limitations.append(f"{distribution} não está instalado neste ambiente")
    determinism = None
    if determinism_report is not None:
        determinism = dict(determinism_report)
        json_value(determinism, "environment.determinism")
    else:
        limitations.append(
            "configure_determinism não executado neste processo: threads e "
            "determinismo não verificados"
        )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "device": "cpu",
        "determinism": determinism,
        "packages": packages,
        "limitations": limitations,
    }


def build_provenance(repo_root: str | Path, *, dataset_path: str | Path,
                     data_record_path: str | Path, execution_spec_path: str | Path,
                     training_seed: int, configuration: Mapping,
                     source_files: Sequence[str] | None = None,
                     determinism_report: Mapping | None = None) -> dict:
    """Constrói identidade de execução sem criar resultado ou artefato científico.

    ``source_files`` (relativos à raiz) participam da identidade científica por
    SHA-256; o default cobre todos os arquivos ``.py`` executados atualmente
    (``research.benchmark``, ``research.data``, ``contracts`` e ``research``).
    Execuções E3.2 devem passar ``source_files`` acrescido dos novos módulos
    executados (FedAvg, runner e demais modelos); alterar qualquer
    dependência listada muda o ``run_id``. O estado dirty e o ambiente ficam em
    ``operational`` e não alteram o ``run_id``.
    """
    _require(type(training_seed) is int and training_seed in TRAINING_SEEDS,
             "training_seed deve pertencer a 42–46")
    json_value(dict(configuration), "configuration")
    root = Path(repo_root).resolve()
    _require(root.is_dir(), "raiz do repositório inexistente")
    dataset_path = Path(dataset_path).resolve()
    data_record_path = Path(data_record_path).resolve()
    execution_spec_path = Path(execution_spec_path).resolve()
    data_record = load_record(data_record_path)
    specification = strict_json(execution_spec_path.read_text(encoding="utf-8"))
    validate_protocol(specification)
    _require(specification["protocol_id"] == "benchmark-v1-execution-1",
             "execution_spec_id inesperado")
    dataset_sha256 = _sha256(dataset_path)
    _require(dataset_sha256 == data_record["dataset"]["sha256"], "dataset SHA-256 divergente")
    selected_files = DEFAULT_SOURCE_FILES if source_files is None else tuple(source_files)
    _require(bool(selected_files), "source_files vazio")
    file_hashes = {}
    for entry in selected_files:
        _require(type(entry) is str and bool(entry), "source_file deve ser caminho relativo")
        relative = _safe_relative(root, root / entry)
        file_hashes[relative] = _sha256(root / relative)
    commit = _git(root, "rev-parse", "HEAD")
    _require(len(commit) == 40, "commit Git inválido")
    identity = {
        "protocol_id": "benchmark-v1",
        "execution_spec_id": specification["protocol_id"],
        "execution_spec_sha256": _sha256(execution_spec_path),
        "dataset_sha256": dataset_sha256,
        "split_id": data_record["split_id"],
        "partition_id": data_record["partition_id"],
        "preprocessing_id": data_record["preprocessing"]["id"],
        "training_seed": training_seed,
        "source_commit": commit,
        "source_files": file_hashes,
        "configuration": dict(configuration),
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    status_lines = [line for line in
                    _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
                    if line.strip()]
    operational = {
        "source_dirty": bool(status_lines),
        "source_status": status_lines,
        "environment": environment_snapshot(determinism_report),
    }
    return {
        "run_id": f"benchmark-{digest[:20]}",
        "scientific_identity": identity,
        "operational": operational,
    }


def artifact_id(run_id: str, artifact_type: str, *, scope: str | None = None,
                client_id: str | None = None) -> str:
    _require(type(run_id) is str and bool(run_id), "run_id inválido")
    _require(type(artifact_type) is str and bool(artifact_type), "artifact_type inválido")
    payload = {
        "run_id": run_id,
        "artifact_type": artifact_type,
        "scope": scope,
        "client_id": client_id,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"{artifact_type}-{digest[:20]}"


def file_reference(repo_root: str | Path, path: str | Path) -> dict:
    """Referência a arquivo existente dentro da raiz, com caminho relativo seguro."""
    root = Path(repo_root).resolve()
    _require(root.is_dir(), "raiz do repositório inexistente")
    resolved = _inside_root(root, Path(path))
    return {"path": _safe_relative(root, resolved), "sha256": _sha256(resolved)}
