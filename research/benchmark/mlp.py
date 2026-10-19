"""Componentes determinísticos das MLPs da E3.1B.2, sem benchmark oficial.

Este módulo materializa a arquitetura congelada ``18→16→8→1`` (449
parâmetros), treino em logits com ``BCEWithLogitsLoss`` e Adam reiniciado no
início de cada bloco. O mesmo ``state_dict`` inicial, derivado pela stream
``mlp_initialization/paired_mlp``, pode ser clonado para a MLP centralizada e
para as três MLPs locais. Estados de optimizer nunca são clonados ou mantidos
entre blocos.

Decisões técnicas: tensores usam ``float32``; a inicialização default dos
``Linear`` ocorre dentro de ``torch.random.fork_rng`` para não alterar o RNG
global; cada época usa uma permutação produzida por ``random.Random`` com a
seed SHA-256 da especificação; Sigmoid aparece somente em ``predict_view``.
PyTorch é importado de forma lazy para que a infraestrutura continue
importável no ambiente stdlib sem a dependência.

Os argumentos de agenda são parametrizáveis para testes unitários pequenos.
Uma execução oficial deve usar os defaults congelados e a E3.2 deverá conferir
o relatório, o escopo de índices e as cinco seeds oficiais antes de publicar
qualquer resultado. FedAvg, validação por bloco, checkpoints e artefatos
oficiais permanecem fora deste módulo e desta etapa.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping

from research.benchmark.data import DatasetView
from research.benchmark.seeding import (
    TRAINING_SEEDS,
    paired_initialization_seed,
    shuffle_seed,
)

BLOCKS = 10
EPOCHS_PER_BLOCK = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_TRAINING_SEED = 2**32 - 1
CLIENT_IDS = ("client-0", "client-1", "client-2")


class MLPError(ValueError):
    """Entrada, estado ou ambiente incompatível com a MLP congelada."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MLPError(message)


@dataclass(frozen=True)
class MLPArchitecture:
    """Dimensões congeladas e contagem analítica incluindo biases."""

    input_features: int = 18
    hidden_1: int = 16
    hidden_2: int = 8
    output_features: int = 1

    @property
    def parameter_count(self) -> int:
        return (
            (self.input_features + 1) * self.hidden_1
            + (self.hidden_1 + 1) * self.hidden_2
            + (self.hidden_2 + 1) * self.output_features
        )


ARCHITECTURE = MLPArchitecture()


@dataclass(frozen=True)
class MLPBatchPrediction:
    """Predições individuais, prontas para ``evaluate_scores/global``."""

    indices: tuple[int, ...]
    labels: tuple[int, ...]
    logits: tuple[float, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class BlockTrainingReport:
    """Escopo e custo medido/teórico de um treino por blocos."""

    model_family: str
    namespace: str
    client_id: str | None
    training_seed: int
    blocks: int
    epochs_per_block: int
    batch_size: int
    learning_rate: float
    examples_processed: int
    optimizer_steps: int
    theoretical_examples: int
    theoretical_steps: int
    fit_indices: tuple[int, ...]


@dataclass(frozen=True)
class TrainingStepReport:
    """Escopo e custo de um único bloco/round com Adam independente."""

    namespace: str
    client_id: str | None
    training_seed: int
    step: int
    epochs: int
    batch_size: int
    learning_rate: float
    examples_processed: int
    optimizer_steps: int
    theoretical_examples: int
    theoretical_steps: int
    fit_indices: tuple[int, ...]


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise MLPError("PyTorch não instalado neste ambiente") from exc
    return torch


def _positive_int(name: str, value: int) -> int:
    _require(type(value) is int and value > 0, f"{name} deve ser inteiro positivo")
    return value


def _training_seed(value: int) -> int:
    _require(type(value) is int and 0 <= value <= MAX_TRAINING_SEED,
             "training_seed deve ser inteiro em [0, 2**32-1]")
    return value


def _namespace(model_family: str, client_id: str | None) -> str:
    _require(model_family in {"mlp_centralized", "mlp_local"},
             "model_family deve ser mlp_centralized ou mlp_local")
    if model_family == "mlp_centralized":
        _require(client_id is None, "mlp_centralized não possui client_id")
        return "central"
    _require(client_id in CLIENT_IDS,
             "mlp_local exige client_id entre client-0, client-1 e client-2")
    return "local"


def _validated_view(view: DatasetView, *, name: str) -> DatasetView:
    _require(isinstance(view, DatasetView), f"{name} deve ser DatasetView")
    _require(len(view) > 0, f"{name} vazio")
    _require(all(len(row) == ARCHITECTURE.input_features for row in view.features),
             f"{name} deve possuir exatamente 18 features por exemplo")
    _require(all(type(label) is int and label in (0, 1) for label in view.labels),
             f"{name}: labels devem ser inteiros 0/1")
    return view


def _validated_network(network, torch):
    _require(isinstance(network, torch.nn.Sequential),
             "network deve ser torch.nn.Sequential")
    expected_types = (
        torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear,
        torch.nn.ReLU, torch.nn.Linear,
    )
    _require(len(network) == len(expected_types)
             and all(type(module) is expected
                     for module, expected in zip(network, expected_types)),
             "network deve seguir Linear/ReLU/Linear/ReLU/Linear")
    linears = [network[0], network[2], network[4]]
    dimensions = [(layer.in_features, layer.out_features) for layer in linears]
    expected = [
        (ARCHITECTURE.input_features, ARCHITECTURE.hidden_1),
        (ARCHITECTURE.hidden_1, ARCHITECTURE.hidden_2),
        (ARCHITECTURE.hidden_2, ARCHITECTURE.output_features),
    ]
    _require(dimensions == expected,
             "network deve seguir exatamente a arquitetura 18→16→8→1")
    _require(sum(parameter.numel() for parameter in network.parameters())
             == ARCHITECTURE.parameter_count,
             "network deve possuir exatamente 449 parâmetros")
    _require(all(parameter.requires_grad for parameter in network.parameters()),
             "os 449 parâmetros da network devem ser treináveis")
    return network


def build_network():
    """Cria a arquitetura congelada; a última camada devolve logits."""
    torch = _torch()
    with torch.random.fork_rng(devices=[]):
        return torch.nn.Sequential(
            torch.nn.Linear(ARCHITECTURE.input_features, ARCHITECTURE.hidden_1),
            torch.nn.ReLU(),
            torch.nn.Linear(ARCHITECTURE.hidden_1, ARCHITECTURE.hidden_2),
            torch.nn.ReLU(),
            torch.nn.Linear(ARCHITECTURE.hidden_2, ARCHITECTURE.output_features),
        )


def build_loss():
    """BCE em logits, sem ``weight`` ou ``pos_weight``."""
    return _torch().nn.BCEWithLogitsLoss()


def clone_state_dict(state: Mapping) -> dict:
    """Cópia tensor-a-tensor sem aliases com o modelo ou outras cópias."""
    torch = _torch()
    _require(isinstance(state, Mapping) and bool(state), "state_dict inválido ou vazio")
    result = {}
    for name, tensor in state.items():
        _require(type(name) is str and bool(name), "nome inválido no state_dict")
        _require(torch.is_tensor(tensor), f"state_dict[{name!r}] não é tensor")
        result[name] = tensor.detach().clone()
    return result


def initial_state_dict(training_seed: int) -> dict:
    """Cria uma única inicialização emparelhada para uma seed oficial."""
    derived_seed = paired_initialization_seed(training_seed)
    torch = _torch()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derived_seed)
        network = build_network()
    return clone_state_dict(network.state_dict())


def _validated_initial_state(initial_state: Mapping, training_seed: int, torch) -> dict:
    """Copia o estado e confere o pareamento exigido para seeds oficiais."""
    _require(isinstance(initial_state, Mapping),
             "initial_state deve ser Mapping")
    state = clone_state_dict(initial_state)
    if training_seed not in TRAINING_SEEDS:
        return state

    expected = initial_state_dict(training_seed)
    same_structure = (
        state.keys() == expected.keys()
        and all(
            state[name].shape == expected[name].shape
            for name in state
        )
    )
    if same_structure:
        paired = all(
            state[name].dtype == expected[name].dtype
            and torch.equal(state[name].detach().cpu(), expected[name])
            for name in state
        )
        _require(
            paired,
            "initial_state não corresponde à inicialização emparelhada de training_seed",
        )
    return state


def final_state(network) -> dict:
    """Snapshot final desacoplado dos parâmetros vivos do modelo."""
    torch = _torch()
    network = _validated_network(network, torch)
    return clone_state_dict(network.state_dict())


def theoretical_costs(example_count: int, *, blocks: int = BLOCKS,
                      epochs_per_block: int = EPOCHS_PER_BLOCK,
                      batch_size: int = BATCH_SIZE) -> tuple[int, int]:
    """Retorna ``(exposições, optimizer_steps)`` com ``drop_last=false``."""
    _require(type(example_count) is int and example_count >= 0,
             "example_count deve ser inteiro não negativo")
    blocks = _positive_int("blocks", blocks)
    epochs_per_block = _positive_int("epochs_per_block", epochs_per_block)
    batch_size = _positive_int("batch_size", batch_size)
    epochs = blocks * epochs_per_block
    return example_count * epochs, math.ceil(example_count / batch_size) * epochs


def epoch_permutation(example_count: int, *, training_seed: int, namespace: str,
                      client_id: str | None, block: int, epoch: int,
                      shuffle_seed_fn=shuffle_seed) -> tuple[int, ...]:
    """Permutação local e determinística para uma época, sem tocar RNG global."""
    _require(type(example_count) is int and example_count >= 0,
             "example_count deve ser inteiro não negativo")
    _training_seed(training_seed)
    _require(namespace in {"central", "federated", "local"},
             "namespace de shuffle inválido")
    if namespace == "central":
        _require(client_id is None, "shuffle central não possui client_id")
    else:
        _require(client_id in CLIENT_IDS,
                 "shuffle federated/local exige client_id congelado")
    _require(type(block) is int and block >= 0, "block deve ser inteiro não negativo")
    _require(type(epoch) is int and epoch >= 0, "epoch deve ser inteiro não negativo")
    _require(callable(shuffle_seed_fn), "shuffle_seed_fn deve ser chamável")
    derived_seed = shuffle_seed_fn(
        training_seed, namespace, client_id=client_id, step=block, epoch=epoch,
    )
    _require(type(derived_seed) is int and derived_seed >= 0,
             "shuffle_seed_fn deve devolver inteiro não negativo")
    order = list(range(example_count))
    random.Random(derived_seed).shuffle(order)
    return tuple(order)


def train_step(network, train_view: DatasetView, *, namespace: str,
               client_id: str | None, training_seed: int, step: int,
               epochs: int = EPOCHS_PER_BLOCK, batch_size: int = BATCH_SIZE,
               learning_rate: float = LEARNING_RATE,
               shuffle_seed_fn=shuffle_seed) -> TrainingStepReport:
    """Treina um bloco/round com Adam novo sobre uma rede já carregada.

    O caller controla o estado inicial da ``network``; o pareamento oficial
    continua sendo validado por ``train_blocks`` antes do primeiro bloco.
    """
    train_view = _validated_view(train_view, name="train_view")
    training_seed = _training_seed(training_seed)
    _require(namespace in {"central", "federated", "local"},
             "namespace de treino inválido")
    if namespace == "central":
        _require(client_id is None, "treino central não possui client_id")
    else:
        _require(client_id in CLIENT_IDS,
                 "treino federated/local exige client_id congelado")
    _require(type(step) is int and step >= 0,
             "step deve ser inteiro não negativo")
    epochs = _positive_int("epochs", epochs)
    batch_size = _positive_int("batch_size", batch_size)
    _require(type(learning_rate) in (int, float)
             and math.isfinite(float(learning_rate)) and learning_rate > 0,
             "learning_rate deve ser número finito positivo")
    _require(callable(shuffle_seed_fn), "shuffle_seed_fn deve ser chamável")

    torch = _torch()
    network = _validated_network(network, torch)
    features = torch.tensor(train_view.features, dtype=torch.float32)
    labels = torch.tensor(train_view.labels, dtype=torch.float32)
    criterion = build_loss()
    optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate))
    examples_processed = 0
    optimizer_steps = 0
    network.train()

    for epoch in range(epochs):
        order = epoch_permutation(
            len(train_view), training_seed=training_seed,
            namespace=namespace, client_id=client_id,
            block=step, epoch=epoch, shuffle_seed_fn=shuffle_seed_fn,
        )
        order_tensor = torch.tensor(order, dtype=torch.long)
        shuffled_features = features[order_tensor]
        shuffled_labels = labels[order_tensor]
        for start in range(0, len(train_view), batch_size):
            batch_features = shuffled_features[start:start + batch_size]
            batch_labels = shuffled_labels[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_features).reshape(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            examples_processed += int(batch_labels.shape[0])
            optimizer_steps += 1

    theoretical_examples, theoretical_steps = theoretical_costs(
        len(train_view), blocks=1, epochs_per_block=epochs,
        batch_size=batch_size,
    )
    return TrainingStepReport(
        namespace=namespace,
        client_id=client_id,
        training_seed=training_seed,
        step=step,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        examples_processed=examples_processed,
        optimizer_steps=optimizer_steps,
        theoretical_examples=theoretical_examples,
        theoretical_steps=theoretical_steps,
        fit_indices=tuple(train_view.indices),
    )


def train_blocks(network, train_view: DatasetView, *, model_family: str,
                 training_seed: int, initial_state: Mapping,
                 client_id: str | None = None,
                 blocks: int = BLOCKS, epochs_per_block: int = EPOCHS_PER_BLOCK,
                 batch_size: int = BATCH_SIZE,
                 learning_rate: float = LEARNING_RATE,
                 shuffle_seed_fn=shuffle_seed) -> BlockTrainingReport:
    """Treina central ou localmente, recriando Adam no início de cada bloco.

    ``initial_state`` deve ser uma cópia da saída única de
    ``initial_state_dict`` no caminho oficial. A função muta ``network`` e
    devolve somente contabilização e prova do escopo de treino.
    """
    train_view = _validated_view(train_view, name="train_view")
    namespace = _namespace(model_family, client_id)
    training_seed = _training_seed(training_seed)
    blocks = _positive_int("blocks", blocks)
    epochs_per_block = _positive_int("epochs_per_block", epochs_per_block)
    batch_size = _positive_int("batch_size", batch_size)
    _require(type(learning_rate) in (int, float)
             and math.isfinite(float(learning_rate)) and learning_rate > 0,
             "learning_rate deve ser número finito positivo")
    _require(callable(shuffle_seed_fn), "shuffle_seed_fn deve ser chamável")

    torch = _torch()
    network = _validated_network(network, torch)
    validated_initial_state = _validated_initial_state(
        initial_state, training_seed, torch,
    )
    try:
        network.load_state_dict(validated_initial_state, strict=True)
    except RuntimeError as exc:
        raise MLPError("initial_state incompatível com a arquitetura") from exc

    examples_processed = 0
    optimizer_steps = 0

    for block in range(blocks):
        step_report = train_step(
            network, train_view, namespace=namespace, client_id=client_id,
            training_seed=training_seed, step=block, epochs=epochs_per_block,
            batch_size=batch_size, learning_rate=float(learning_rate),
            shuffle_seed_fn=shuffle_seed_fn,
        )
        examples_processed += step_report.examples_processed
        optimizer_steps += step_report.optimizer_steps

    theoretical_examples, theoretical_steps = theoretical_costs(
        len(train_view), blocks=blocks, epochs_per_block=epochs_per_block,
        batch_size=batch_size,
    )
    return BlockTrainingReport(
        model_family=model_family,
        namespace=namespace,
        client_id=client_id,
        training_seed=training_seed,
        blocks=blocks,
        epochs_per_block=epochs_per_block,
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        examples_processed=examples_processed,
        optimizer_steps=optimizer_steps,
        theoretical_examples=theoretical_examples,
        theoretical_steps=theoretical_steps,
        fit_indices=tuple(train_view.indices),
    )


def predict_view(network, view: DatasetView) -> MLPBatchPrediction:
    """Infere em ordem do ``DatasetView`` e restaura o modo anterior da rede."""
    view = _validated_view(view, name="view")
    torch = _torch()
    network = _validated_network(network, torch)
    was_training = network.training
    network.eval()
    try:
        with torch.no_grad():
            features = torch.tensor(view.features, dtype=torch.float32)
            tensor_logits = network(features).reshape(-1)
            tensor_scores = torch.sigmoid(tensor_logits)
            logits = tuple(float(value) for value in tensor_logits)
            scores = tuple(float(value) for value in tensor_scores)
    finally:
        network.train(was_training)
    _require(all(math.isfinite(value) for value in logits),
             "network produziu logits não finitos")
    _require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores),
             "network produziu scores inválidos")
    return MLPBatchPrediction(
        indices=tuple(view.indices), labels=tuple(view.labels),
        logits=logits, scores=scores,
    )
