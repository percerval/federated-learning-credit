"""Derivação determinística dos subconjuntos simulados ``client_test``."""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from research.benchmark.data import DatasetView
from research.benchmark.seeding import canonical_json_bytes


EXECUTION_SPEC_ID = "benchmark-v1-execution-1"
PARTITION_SEED = 44
CLIENT_IDS = ("client-0", "client-1", "client-2")
STREAM = "client_test_partition"


class ClientTestError(ValueError):
    """Test global incompatível com a partição ``client_test`` congelada."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClientTestError(message)


def _partition_digest(index: int, label: int) -> bytes:
    payload = {
        "execution_spec_id": EXECUTION_SPEC_ID,
        "partition_seed": PARTITION_SEED,
        "stream": STREAM,
        "label": label,
        "index": index,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).digest()


def _validate_test_view(test: DatasetView) -> None:
    _require(isinstance(test, DatasetView), "test deve ser DatasetView")
    _require(len(test) > 0, "test vazio")
    _require(
        all(type(index) is int and index >= 0 for index in test.indices),
        "test possui índice inválido",
    )
    _require(len(set(test.indices)) == len(test), "test possui índices duplicados")
    _require(
        all(type(label) is int and label in (0, 1) for label in test.labels),
        "test possui label diferente de inteiro 0/1",
    )
    _require(all(type(row) is tuple for row in test.features),
             "test possui linha de features mutável ou malformada")
    widths = {len(row) for row in test.features}
    _require(len(widths) == 1 and next(iter(widths)) > 0,
             "test possui linhas de features vazias ou com larguras divergentes")
    _require(
        all(
            type(value) is not bool
            and isinstance(value, Real)
            and math.isfinite(float(value))
            for row in test.features for value in row
        ),
        "test possui feature não numérica ou não finita",
    )


def derive_client_tests(test: DatasetView) -> Mapping[str, DatasetView]:
    """Particiona somente o test global pela regra congelada da execution spec.

    Para cada label observado, ordena os exemplos por
    ``(SHA-256(canonical_json(payload)), index)`` e envia a posição ``k`` para
    ``client-(k mod 3)``. As views devolvidas são reordenadas pelo índice
    original e não acessam treino, validation ou qualquer fonte de aleatoriedade.
    """
    _validate_test_view(test)
    records = {
        index: (features, label)
        for index, features, label in zip(test.indices, test.features, test.labels)
    }
    assigned: dict[str, list[int]] = {client_id: [] for client_id in CLIENT_IDS}

    for label in (0, 1):
        ordered = sorted(
            (index for index in test.indices if records[index][1] == label),
            key=lambda index: (_partition_digest(index, label), index),
        )
        for position, index in enumerate(ordered):
            assigned[CLIENT_IDS[position % len(CLIENT_IDS)]].append(index)

    views = {}
    covered: set[int] = set()
    label_counts = {label: [] for label in (0, 1)}
    for client_id in CLIENT_IDS:
        indices = tuple(sorted(assigned[client_id]))
        current = set(indices)
        _require(not covered.intersection(current),
                 f"subconjuntos client_test sobrepostos em {client_id}")
        covered.update(current)
        views[client_id] = DatasetView(
            indices=indices,
            features=tuple(tuple(records[index][0]) for index in indices),
            labels=tuple(records[index][1] for index in indices),
        )
        for label in (0, 1):
            label_counts[label].append(sum(value == label for value in views[client_id].labels))

    expected = set(test.indices)
    _require(covered == expected, "client_test não cobre exatamente o test global")
    _require(sum(len(view) for view in views.values()) == len(test),
             "contagem total de client_test divergente")
    for label, counts in label_counts.items():
        _require(max(counts) - min(counts) <= 1,
                 f"distribuição estratificada inválida para label {label}")

    return MappingProxyType(views)
