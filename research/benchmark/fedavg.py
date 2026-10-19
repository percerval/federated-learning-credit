"""Agregação e coordenação FedAvg determinísticas, sem benchmark oficial."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from research.benchmark.data import DatasetView
from research.benchmark.evaluation import evaluate_scores
from research.benchmark.mlp import (
    BATCH_SIZE,
    BLOCKS,
    CLIENT_IDS,
    EPOCHS_PER_BLOCK,
    LEARNING_RATE,
    _training_seed,
    _validated_initial_state,
    _validated_view,
    build_network,
    clone_state_dict,
    final_state,
    predict_view,
    train_step,
)
from research.benchmark.seeding import (
    TRAINING_SEEDS,
    paired_initialization_seed,
    shuffle_seed,
)


NO_SERIALIZATION_NOTE = (
    "zero indica ausência de serialização no coordenador in-process; "
    "não representa tráfego de rede medido"
)


class FedAvgError(ValueError):
    """Estados locais ou pesos incompatíveis com a agregação congelada."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FedAvgError(message)


@dataclass(frozen=True)
class FederatedClientRoundReport:
    """Treino de um cliente em um round, sem estado de optimizer persistido."""

    round_index: int
    client_id: str
    training_seed: int
    example_count: int
    aggregation_weight: float
    shuffle_seeds: tuple[int, ...]
    fit_indices: tuple[int, ...]
    examples_processed: int
    optimizer_steps: int
    theoretical_examples: int
    theoretical_steps: int


@dataclass(frozen=True)
class FederatedRoundReport:
    """Resultado imutável de treino, agregação, comunicação e validation."""

    round_index: int
    clients: tuple[FederatedClientRoundReport, ...]
    examples_processed: int
    optimizer_steps: int
    theoretical_examples: int
    theoretical_steps: int
    validation_indices: tuple[int, ...]
    validation_metrics: Mapping
    logical_downloads: int
    logical_uploads: int
    theoretical_download_bytes: int
    theoretical_upload_bytes: int
    serialized_download_bytes: int
    serialized_upload_bytes: int
    serialization_note: str


@dataclass(frozen=True)
class FedAvgTrainingReport:
    """Configuração, proveniência de seeds e custos do treinamento federado."""

    training_seed: int
    initialization_seed: int | None
    rounds: int
    epochs_per_round: int
    batch_size: int
    learning_rate: float
    client_ids: tuple[str, ...]
    round_reports: tuple[FederatedRoundReport, ...]
    examples_processed: int
    optimizer_steps: int
    theoretical_examples: int
    theoretical_steps: int


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise FedAvgError("PyTorch não instalado neste ambiente") from exc
    return torch


def _positive_int(name: str, value: int) -> int:
    _require(type(value) is int and value > 0,
             f"{name} deve ser inteiro positivo")
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _require_official_clients(mapping: Mapping, *, name: str) -> None:
    _require(isinstance(mapping, Mapping), f"{name} deve ser Mapping")
    clients = set(mapping)
    expected = set(CLIENT_IDS)
    missing = expected - clients
    extra = clients - expected
    _require(
        not missing and not extra,
        f"{name} deve conter exatamente {CLIENT_IDS}; "
        f"ausentes={sorted(map(repr, missing))!r}, "
        f"extras={sorted(map(repr, extra))!r}",
    )


def aggregate_state_dicts(local_states: Mapping[str, Mapping],
                          example_counts: Mapping[str, int]) -> dict:
    """Calcula FedAvg ponderado em ``CLIENT_IDS``, sem mutar nem criar aliases."""
    _require_official_clients(local_states, name="local_states")
    _require_official_clients(example_counts, name="example_counts")
    for client_id in CLIENT_IDS:
        count = example_counts[client_id]
        _require(type(count) is int and count > 0,
                 f"example_counts[{client_id!r}] deve ser inteiro positivo")

    torch = _torch()
    reference = local_states[CLIENT_IDS[0]]
    _require(isinstance(reference, Mapping) and bool(reference),
             f"estado de {CLIENT_IDS[0]} inválido ou vazio")
    parameter_names = tuple(reference.keys())
    _require(all(type(name) is str and bool(name) for name in parameter_names),
             "nomes de tensors devem ser strings não vazias")

    expected_names = set(parameter_names)
    for client_id in CLIENT_IDS:
        state = local_states[client_id]
        _require(isinstance(state, Mapping) and bool(state),
                 f"estado de {client_id} inválido ou vazio")
        _require(set(state) == expected_names,
                 f"chaves do estado de {client_id} divergentes")
        for name in parameter_names:
            tensor = state[name]
            _require(torch.is_tensor(tensor),
                     f"estado de {client_id}[{name!r}] não é tensor")
            _require(tensor.dtype == torch.float32,
                     f"estado de {client_id}[{name!r}] deve usar float32")
            _require(tensor.shape == reference[name].shape,
                     f"shape de {client_id}[{name!r}] divergente")
            _require(bool(torch.isfinite(tensor).all().item()),
                     f"estado de {client_id}[{name!r}] contém NaN/Inf")

    total_examples = sum(example_counts[client_id] for client_id in CLIENT_IDS)
    aggregated = {}
    for name in parameter_names:
        result = torch.zeros_like(reference[name], dtype=torch.float32)
        for client_id in CLIENT_IDS:
            weight = example_counts[client_id] / total_examples
            result.add_(local_states[client_id][name].detach(), alpha=weight)
        _require(bool(torch.isfinite(result).all().item()),
                 f"agregação de {name!r} produziu NaN/Inf")
        aggregated[name] = result
    return aggregated


def _load_state(network, state: Mapping, *, name: str) -> None:
    try:
        network.load_state_dict(clone_state_dict(state), strict=True)
    except RuntimeError as exc:
        raise FedAvgError(f"{name} incompatível com a arquitetura") from exc


def train_fedavg(network, clients: Mapping[str, DatasetView],
                 validation: DatasetView, *, training_seed: int,
                 initial_state: Mapping, rounds: int = BLOCKS,
                 epochs_per_round: int = EPOCHS_PER_BLOCK,
                 batch_size: int = BATCH_SIZE,
                 learning_rate: float = LEARNING_RATE,
                 shuffle_seed_fn=shuffle_seed) -> FedAvgTrainingReport:
    """Treina e modifica a MLP global com participação 3/3.

    Em caso de falha durante uma rodada, ``network`` conserva o estado global
    do último round concluído. Uma nova chamada recarrega ``initial_state``,
    permitindo um retry determinístico. Validation permanece não adaptativa.
    """
    _require_official_clients(clients, name="clients")
    training_seed = _training_seed(training_seed)
    rounds = _positive_int("rounds", rounds)
    epochs_per_round = _positive_int("epochs_per_round", epochs_per_round)
    batch_size = _positive_int("batch_size", batch_size)
    _require(type(learning_rate) in (int, float)
             and math.isfinite(float(learning_rate)) and learning_rate > 0,
             "learning_rate deve ser número finito positivo")
    _require(callable(shuffle_seed_fn), "shuffle_seed_fn deve ser chamável")

    checked_clients = {
        client_id: _validated_view(clients[client_id], name=f"clients[{client_id!r}]")
        for client_id in CLIENT_IDS
    }
    validation = _validated_view(validation, name="validation")
    covered_indices: set[int] = set()
    for client_id in CLIENT_IDS:
        _require(all(type(index) is int for index in checked_clients[client_id].indices),
                 f"clients[{client_id!r}] possui índice inválido")
        client_indices = set(checked_clients[client_id].indices)
        _require(len(client_indices) == len(checked_clients[client_id]),
                 f"clients[{client_id!r}] possui índices duplicados")
        _require(not covered_indices.intersection(client_indices),
                 f"clients[{client_id!r}] sobrepõe outro cliente")
        covered_indices.update(client_indices)
    _require(all(type(index) is int for index in validation.indices),
             "validation possui índice inválido")
    _require(len(set(validation.indices)) == len(validation),
             "validation possui índices duplicados")
    _require(validation.indices == tuple(sorted(validation.indices)),
             "validation deve preservar ordem crescente dos índices")
    _require(not covered_indices.intersection(validation.indices),
             "clientes de treino intersectam validation")

    torch = _torch()
    final_state(network)
    validated_initial_state = _validated_initial_state(
        initial_state, training_seed, torch,
    )
    _load_state(network, validated_initial_state, name="initial_state")
    global_state = final_state(network)
    model_parameter_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in global_state.values()
    )
    example_counts = {
        client_id: len(checked_clients[client_id])
        for client_id in CLIENT_IDS
    }
    total_examples = sum(example_counts.values())
    round_reports = []
    examples_processed = 0
    optimizer_steps = 0

    for round_index in range(rounds):
        local_states = {}
        client_reports = []
        for client_id in CLIENT_IDS:
            local_network = build_network()
            _load_state(local_network, global_state, name="global_state")
            consumed_shuffle_seeds = []

            def recording_shuffle_seed(seed, namespace, *, client_id, step, epoch):
                derived = shuffle_seed_fn(
                    seed, namespace, client_id=client_id, step=step, epoch=epoch,
                )
                consumed_shuffle_seeds.append(derived)
                return derived

            step_report = train_step(
                local_network, checked_clients[client_id],
                namespace="federated", client_id=client_id,
                training_seed=training_seed, step=round_index,
                epochs=epochs_per_round, batch_size=batch_size,
                learning_rate=float(learning_rate),
                shuffle_seed_fn=recording_shuffle_seed,
            )
            local_states[client_id] = final_state(local_network)
            client_reports.append(FederatedClientRoundReport(
                round_index=round_index,
                client_id=client_id,
                training_seed=training_seed,
                example_count=example_counts[client_id],
                aggregation_weight=example_counts[client_id] / total_examples,
                shuffle_seeds=tuple(consumed_shuffle_seeds),
                fit_indices=step_report.fit_indices,
                examples_processed=step_report.examples_processed,
                optimizer_steps=step_report.optimizer_steps,
                theoretical_examples=step_report.theoretical_examples,
                theoretical_steps=step_report.theoretical_steps,
            ))
            examples_processed += step_report.examples_processed
            optimizer_steps += step_report.optimizer_steps

        aggregated_state = aggregate_state_dicts(local_states, example_counts)
        _load_state(network, aggregated_state, name="aggregated_state")
        global_state = final_state(network)
        prediction = predict_view(network, validation)
        validation_metrics = evaluate_scores(
            prediction.labels, prediction.scores, logits=prediction.logits,
        )
        round_reports.append(FederatedRoundReport(
            round_index=round_index,
            clients=tuple(client_reports),
            examples_processed=sum(
                client.examples_processed for client in client_reports
            ),
            optimizer_steps=sum(
                client.optimizer_steps for client in client_reports
            ),
            theoretical_examples=sum(
                client.theoretical_examples for client in client_reports
            ),
            theoretical_steps=sum(
                client.theoretical_steps for client in client_reports
            ),
            validation_indices=prediction.indices,
            validation_metrics=_freeze(validation_metrics),
            logical_downloads=len(CLIENT_IDS),
            logical_uploads=len(CLIENT_IDS),
            theoretical_download_bytes=model_parameter_bytes * len(CLIENT_IDS),
            theoretical_upload_bytes=model_parameter_bytes * len(CLIENT_IDS),
            serialized_download_bytes=0,
            serialized_upload_bytes=0,
            serialization_note=NO_SERIALIZATION_NOTE,
        ))

    theoretical_examples = sum(
        client.theoretical_examples
        for report in round_reports for client in report.clients
    )
    theoretical_steps = sum(
        client.theoretical_steps
        for report in round_reports for client in report.clients
    )
    return FedAvgTrainingReport(
        training_seed=training_seed,
        initialization_seed=(
            paired_initialization_seed(training_seed)
            if training_seed in TRAINING_SEEDS else None
        ),
        rounds=rounds,
        epochs_per_round=epochs_per_round,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        client_ids=CLIENT_IDS,
        round_reports=tuple(round_reports),
        examples_processed=examples_processed,
        optimizer_steps=optimizer_steps,
        theoretical_examples=theoretical_examples,
        theoretical_steps=theoretical_steps,
    )
