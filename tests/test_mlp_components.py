"""Testes E3.1B.2 — componentes MLP, sem benchmark ou artefatos oficiais.

Treinos usam somente dados sintéticos, seeds locais e uma função de shuffle
injetada. Os testes podem usar as seeds oficiais 42–46 em derivação,
permutação, inicialização emparelhada e chamadas mínimas a ``train_blocks``
para validar gates e o pareamento ``training_seed``/``initial_state``, mas
nunca executam o benchmark oficial: agenda congelada, dataset real e
artefatos oficiais permanecem fora destes testes. Testes que dependem de
PyTorch são pulados no ambiente stdlib e executados no venv histórico.
"""

import importlib.util
import math
import random
import unittest
from unittest import mock

from research.benchmark import mlp as mlp_module
from research.benchmark.data import DatasetView
from research.benchmark.evaluation import evaluate_global, evaluate_scores
from research.benchmark.mlp import (
    ARCHITECTURE,
    BATCH_SIZE,
    BLOCKS,
    EPOCHS_PER_BLOCK,
    LEARNING_RATE,
    MLPArchitecture,
    MLPError,
    TrainingStepReport,
    build_loss,
    build_network,
    clone_state_dict,
    epoch_permutation,
    final_state,
    initial_state_dict,
    predict_view,
    theoretical_costs,
    train_blocks,
    train_step,
)
from research.benchmark.seeding import SeedingError, TRAINING_SEEDS, shuffle_seed

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _view(count, *, start=10_000, shift=0.0, width=18, labels=None):
    if labels is None:
        labels = tuple(index % 2 for index in range(count))
    features = tuple(
        tuple((((index + column) % 7) - 3) / 3.0 + shift
              for column in range(width))
        for index in range(count)
    )
    return DatasetView(
        indices=tuple(start + index for index in range(count)),
        features=features,
        labels=tuple(labels),
    )


def _synthetic_shuffle_seed(training_seed, namespace, *, client_id, step, epoch):
    namespace_offset = {"central": 0, "federated": 100, "local": 200}[namespace]
    client_offset = 0 if client_id is None else int(client_id[-1]) + 1
    return training_seed * 10_000 + namespace_offset + client_offset * 1000 + step * 10 + epoch


class FrozenMLPContractTests(unittest.TestCase):
    """Contratos aritméticos e de seed que não importam PyTorch."""

    def test_architecture_and_schedule_match_frozen_specification(self):
        self.assertEqual(ARCHITECTURE, MLPArchitecture(18, 16, 8, 1))
        self.assertEqual(ARCHITECTURE.parameter_count, 449)
        self.assertEqual(BLOCKS, 10)
        self.assertEqual(EPOCHS_PER_BLOCK, 5)
        self.assertEqual(BATCH_SIZE, 32)
        self.assertEqual(LEARNING_RATE, 0.01)
        self.assertEqual(TRAINING_SEEDS, (42, 43, 44, 45, 46))

    def test_theoretical_costs_match_frozen_budget(self):
        self.assertEqual(theoretical_costs(3576), (178800, 5600))
        local_examples, local_steps = theoretical_costs(1192)
        self.assertEqual((local_examples, local_steps), (59600, 1900))
        self.assertEqual((3 * local_examples, 3 * local_steps), (178800, 5700))
        self.assertEqual(
            theoretical_costs(100, blocks=1, epochs_per_block=1, batch_size=32),
            (100, 4),
        )

    def test_theoretical_costs_reject_invalid_arguments(self):
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(MLPError):
                theoretical_costs(count)
        for overrides in ({"blocks": 0}, {"epochs_per_block": 0}, {"batch_size": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(MLPError):
                theoretical_costs(10, **overrides)

    def test_epoch_permutation_is_deterministic_and_uses_all_coordinates(self):
        calls = []

        def spy(training_seed, namespace, *, client_id, step, epoch):
            calls.append((training_seed, namespace, client_id, step, epoch))
            return _synthetic_shuffle_seed(
                training_seed, namespace, client_id=client_id, step=step, epoch=epoch,
            )

        first = epoch_permutation(
            40, training_seed=7, namespace="local", client_id="client-1",
            block=2, epoch=3, shuffle_seed_fn=spy,
        )
        second = epoch_permutation(
            40, training_seed=7, namespace="local", client_id="client-1",
            block=2, epoch=3, shuffle_seed_fn=spy,
        )
        changed = epoch_permutation(
            40, training_seed=7, namespace="local", client_id="client-1",
            block=2, epoch=4, shuffle_seed_fn=spy,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(sorted(first), list(range(40)))
        self.assertEqual(calls[0], (7, "local", "client-1", 2, 3))

    def test_real_shuffle_contract_with_official_seed_is_pure(self):
        random.seed(1234)
        expected_next = random.random()
        random.seed(1234)
        actual = epoch_permutation(
            20, training_seed=42, namespace="central", client_id=None,
            block=1, epoch=2,
        )
        next_after_call = random.random()
        derived = shuffle_seed(42, "central", client_id=None, step=1, epoch=2)
        expected = list(range(20))
        random.Random(derived).shuffle(expected)
        self.assertEqual(actual, tuple(expected))
        self.assertEqual(next_after_call, expected_next)

    def test_default_shuffle_gate_rejects_non_official_seed_without_fit(self):
        with self.assertRaises(SeedingError):
            epoch_permutation(
                10, training_seed=7, namespace="central", client_id=None,
                block=0, epoch=0,
            )

    def test_epoch_permutation_rejects_invalid_scope(self):
        invalid = (
            {"namespace": "other", "client_id": None},
            {"namespace": "central", "client_id": "client-0"},
            {"namespace": "local", "client_id": None},
            {"namespace": "local", "client_id": "client-9"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(MLPError):
                epoch_permutation(
                    10, training_seed=7, block=0, epoch=0,
                    shuffle_seed_fn=_synthetic_shuffle_seed, **values,
                )


@unittest.skipUnless(HAS_TORCH, "PyTorch indisponível")
class TorchMLPComponentTests(unittest.TestCase):
    """Treinos curtos e sintéticos; seeds oficiais só validam gates e pareamento."""

    @staticmethod
    def _trained(view, *, seed=7, family="mlp_centralized", client_id=None,
                 blocks=2, epochs=2, batch_size=8):
        state = initial_state_dict(42)
        network = build_network()
        report = train_blocks(
            network, view, model_family=family, client_id=client_id,
            training_seed=seed, blocks=blocks, epochs_per_block=epochs,
            batch_size=batch_size, shuffle_seed_fn=_synthetic_shuffle_seed,
            initial_state=state,
        )
        return network, report

    def test_network_structure_parameter_count_and_loss(self):
        import torch

        network = build_network()
        self.assertIsInstance(network, torch.nn.Sequential)
        self.assertEqual(
            [(module.in_features, module.out_features)
             for module in network if isinstance(module, torch.nn.Linear)],
            [(18, 16), (16, 8), (8, 1)],
        )
        self.assertEqual(
            sum(isinstance(module, torch.nn.ReLU) for module in network), 2)
        self.assertIsInstance(network[-1], torch.nn.Linear)
        self.assertEqual(sum(parameter.numel() for parameter in network.parameters()), 449)
        loss = build_loss()
        self.assertIsInstance(loss, torch.nn.BCEWithLogitsLoss)
        self.assertIsNone(loss.weight)
        self.assertIsNone(loss.pos_weight)

        wrong_activation = torch.nn.Sequential(
            torch.nn.Linear(18, 16), torch.nn.Sigmoid(),
            torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1),
        )
        with self.assertRaises(MLPError):
            final_state(wrong_activation)
        next(network.parameters()).requires_grad_(False)
        with self.assertRaises(MLPError):
            final_state(network)

    def test_build_network_is_deterministic_without_advancing_global_rng(self):
        import torch

        torch.manual_seed(1234)
        rng_before = torch.random.get_rng_state().clone()
        first = build_network()
        rng_after = torch.random.get_rng_state().clone()
        second = build_network()

        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        first_state = first.state_dict()
        second_state = second.state_dict()
        self.assertTrue(all(torch.equal(first_state[key], second_state[key])
                            for key in first_state))

    def test_paired_initialization_is_deterministic_cloned_and_rng_isolated(self):
        import torch

        torch.manual_seed(123)
        expected_next = torch.rand(4)
        torch.manual_seed(123)
        first = initial_state_dict(42)
        actual_next = torch.rand(4)
        second = initial_state_dict(42)
        different_seed = initial_state_dict(43)
        self.assertTrue(torch.equal(expected_next, actual_next))
        self.assertEqual(set(first), set(second))
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))
        self.assertTrue(any(not torch.equal(first[key], different_seed[key])
                            for key in first))

        clone_a = clone_state_dict(first)
        clone_b = clone_state_dict(first)
        key = next(iter(first))
        clone_a[key].add_(1.0)
        self.assertFalse(torch.equal(clone_a[key], first[key]))
        self.assertTrue(torch.equal(clone_b[key], first[key]))
        with self.assertRaises(SeedingError):
            initial_state_dict(7)

    def test_official_training_seed_rejects_unpaired_state_before_training(self):
        import torch

        state = initial_state_dict(42)
        snapshot = clone_state_dict(state)
        network = build_network()
        with mock.patch("torch.optim.Adam") as adam:
            with self.assertRaisesRegex(
                MLPError,
                "initial_state não corresponde.*training_seed",
            ):
                train_blocks(
                    network, _view(8), model_family="mlp_centralized",
                    training_seed=43, initial_state=state,
                    blocks=1, epochs_per_block=1, batch_size=8,
                    shuffle_seed_fn=_synthetic_shuffle_seed,
                )
        adam.assert_not_called()
        self.assertTrue(all(torch.equal(state[key], snapshot[key]) for key in state))

    def test_official_training_seed_accepts_paired_state_and_proceeds(self):
        import torch

        state = initial_state_dict(44)
        network = build_network()
        report = train_blocks(
            network, _view(8), model_family="mlp_centralized",
            training_seed=44, initial_state=state,
            blocks=1, epochs_per_block=1, batch_size=8,
            shuffle_seed_fn=_synthetic_shuffle_seed,
        )
        self.assertEqual(report.training_seed, 44)
        self.assertEqual(report.optimizer_steps, 1)
        self.assertEqual(report.examples_processed, 8)
        self.assertEqual(report.optimizer_steps, report.theoretical_steps)
        trained_state = final_state(network)
        self.assertTrue(any(
            not torch.equal(trained_state[key], state[key])
            for key in trained_state
        ))

    def test_pairing_validation_preserves_global_rng_on_accept_and_reject(self):
        import torch

        original_validation = mlp_module._validated_initial_state
        observed = []

        def spy(initial_state, training_seed, torch_module):
            before = torch.random.get_rng_state().clone()
            try:
                return original_validation(initial_state, training_seed, torch_module)
            finally:
                observed.append(torch.equal(before, torch.random.get_rng_state()))

        with mock.patch.object(mlp_module, "_validated_initial_state", spy):
            accepted = train_blocks(
                build_network(), _view(8), model_family="mlp_centralized",
                training_seed=44, initial_state=initial_state_dict(44),
                blocks=1, epochs_per_block=1, batch_size=8,
                shuffle_seed_fn=_synthetic_shuffle_seed,
            )
            self.assertEqual(accepted.optimizer_steps, 1)

            rejected_network = build_network()
            rejected_state = initial_state_dict(42)
            rng_before_rejection = torch.random.get_rng_state().clone()
            with self.assertRaisesRegex(MLPError, "não corresponde"):
                train_blocks(
                    rejected_network, _view(8), model_family="mlp_centralized",
                    training_seed=43, initial_state=rejected_state,
                    blocks=1, epochs_per_block=1, batch_size=8,
                    shuffle_seed_fn=_synthetic_shuffle_seed,
                )
            self.assertTrue(torch.equal(
                rng_before_rejection, torch.random.get_rng_state(),
            ))

        self.assertEqual(observed, [True, True])

    def test_same_state_data_and_seed_produce_identical_training(self):
        import torch

        view = _view(35)
        first, _ = self._trained(view, seed=7)
        second, _ = self._trained(view, seed=7)
        other, _ = self._trained(view, seed=11)
        first_state = final_state(first)
        second_state = final_state(second)
        other_state = final_state(other)
        self.assertTrue(all(torch.equal(first_state[key], second_state[key])
                            for key in first_state))
        self.assertTrue(any(not torch.equal(first_state[key], other_state[key])
                            for key in first_state))

    def test_adam_is_recreated_for_every_block_with_independent_state(self):
        import torch

        created = []
        real_adam = torch.optim.Adam

        def spy(*args, **kwargs):
            optimizer = real_adam(*args, **kwargs)
            created.append(optimizer)
            return optimizer

        network = build_network()
        with mock.patch("torch.optim.Adam", side_effect=spy):
            report = train_blocks(
                network, _view(13), model_family="mlp_centralized",
                training_seed=7, blocks=3, epochs_per_block=1, batch_size=8,
                shuffle_seed_fn=_synthetic_shuffle_seed,
                initial_state=initial_state_dict(42),
            )
        self.assertEqual(report.blocks, 3)
        self.assertEqual(len(created), 3)
        self.assertEqual(len({id(optimizer) for optimizer in created}), 3)
        self.assertEqual(len({id(optimizer.state) for optimizer in created}), 3)
        created[0].param_groups[0]["lr"] = 999.0
        self.assertEqual(created[1].param_groups[0]["lr"], LEARNING_RATE)

    def test_train_step_uses_explicit_scope_and_independent_adam(self):
        import torch

        calls = []
        created = []
        real_adam = torch.optim.Adam

        def seed_spy(training_seed, namespace, *, client_id, step, epoch):
            calls.append((training_seed, namespace, client_id, step, epoch))
            return _synthetic_shuffle_seed(
                training_seed, namespace, client_id=client_id,
                step=step, epoch=epoch,
            )

        def adam_spy(*args, **kwargs):
            optimizer = real_adam(*args, **kwargs)
            created.append(optimizer)
            return optimizer

        network = build_network()
        network.load_state_dict(initial_state_dict(42), strict=True)
        torch.manual_seed(987)
        rng_before = torch.random.get_rng_state().clone()
        with mock.patch("torch.optim.Adam", side_effect=adam_spy):
            first = train_step(
                network, _view(13), namespace="federated",
                client_id="client-1", training_seed=7, step=4,
                epochs=2, batch_size=8, learning_rate=0.02,
                shuffle_seed_fn=seed_spy,
            )
            second = train_step(
                network, _view(13), namespace="federated",
                client_id="client-1", training_seed=7, step=5,
                epochs=1, batch_size=8, learning_rate=0.02,
                shuffle_seed_fn=seed_spy,
            )

        self.assertIsInstance(first, TrainingStepReport)
        self.assertEqual(
            (first.namespace, first.client_id, first.step, first.epochs),
            ("federated", "client-1", 4, 2),
        )
        self.assertEqual(first.examples_processed, 26)
        self.assertEqual(first.optimizer_steps, 4)
        self.assertEqual(first.examples_processed, first.theoretical_examples)
        self.assertEqual(first.optimizer_steps, first.theoretical_steps)
        self.assertEqual(second.step, 5)
        self.assertEqual(len(created), 2)
        self.assertEqual(len({id(optimizer) for optimizer in created}), 2)
        self.assertEqual(len({id(optimizer.state) for optimizer in created}), 2)
        self.assertEqual(calls, [
            (7, "federated", "client-1", 4, 0),
            (7, "federated", "client-1", 4, 1),
            (7, "federated", "client-1", 5, 0),
        ])
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_train_blocks_matches_explicit_train_steps(self):
        import torch

        view = _view(21)
        state = initial_state_dict(42)
        blocks_network = build_network()
        blocks_report = train_blocks(
            blocks_network, view, model_family="mlp_centralized",
            training_seed=7, initial_state=state,
            blocks=2, epochs_per_block=3, batch_size=8,
            learning_rate=0.02, shuffle_seed_fn=_synthetic_shuffle_seed,
        )

        steps_network = build_network()
        steps_network.load_state_dict(clone_state_dict(state), strict=True)
        step_reports = tuple(
            train_step(
                steps_network, view, namespace="central", client_id=None,
                training_seed=7, step=step, epochs=3, batch_size=8,
                learning_rate=0.02,
                shuffle_seed_fn=_synthetic_shuffle_seed,
            )
            for step in range(2)
        )

        blocks_state = final_state(blocks_network)
        steps_state = final_state(steps_network)
        self.assertTrue(all(torch.equal(blocks_state[key], steps_state[key])
                            for key in blocks_state))
        self.assertEqual(
            blocks_report.examples_processed,
            sum(report.examples_processed for report in step_reports),
        )
        self.assertEqual(
            blocks_report.optimizer_steps,
            sum(report.optimizer_steps for report in step_reports),
        )
        self.assertEqual(blocks_report.fit_indices, view.indices)

    def test_accounting_includes_partial_batch_and_derives_each_epoch_seed(self):
        calls = []

        def spy(training_seed, namespace, *, client_id, step, epoch):
            calls.append((training_seed, namespace, client_id, step, epoch))
            return _synthetic_shuffle_seed(
                training_seed, namespace, client_id=client_id, step=step, epoch=epoch,
            )

        network = build_network()
        report = train_blocks(
            network, _view(35), model_family="mlp_centralized", training_seed=7,
            blocks=2, epochs_per_block=3, batch_size=32,
            shuffle_seed_fn=spy, initial_state=initial_state_dict(42),
        )
        self.assertEqual(report.examples_processed, 35 * 2 * 3)
        self.assertEqual(report.optimizer_steps, 2 * 2 * 3)
        self.assertEqual(report.examples_processed, report.theoretical_examples)
        self.assertEqual(report.optimizer_steps, report.theoretical_steps)
        self.assertEqual(calls, [
            (7, "central", None, block, epoch)
            for block in range(2) for epoch in range(3)
        ])

    def test_three_local_models_are_independent_and_record_exact_fit_scope(self):
        import torch

        state = initial_state_dict(42)
        trained = {}
        reports = {}
        views = {
            client_id: _view(17, start=20_000 + position * 100, shift=position * 0.4)
            for position, client_id in enumerate(("client-0", "client-1", "client-2"))
        }
        for position, (client_id, view) in enumerate(views.items()):
            network = build_network()
            reports[client_id] = train_blocks(
                network, view, model_family="mlp_local", client_id=client_id,
                training_seed=7 + position, blocks=1, epochs_per_block=2,
                batch_size=8, shuffle_seed_fn=_synthetic_shuffle_seed,
                initial_state=state,
            )
            trained[client_id] = final_state(network)
            self.assertEqual(reports[client_id].fit_indices, view.indices)
            self.assertEqual(reports[client_id].namespace, "local")

        rerun, rerun_report = self._trained(
            views["client-0"], seed=7, family="mlp_local", client_id="client-0",
            blocks=1, epochs=2, batch_size=8,
        )
        rerun_state = final_state(rerun)
        self.assertEqual(rerun_report.fit_indices, views["client-0"].indices)
        self.assertTrue(all(torch.equal(trained["client-0"][key], rerun_state[key])
                            for key in rerun_state))
        self.assertTrue(any(not torch.equal(trained["client-0"][key],
                                            trained["client-1"][key])
                            for key in rerun_state))
        all_indices = [set(view.indices) for view in views.values()]
        self.assertTrue(all_indices[0].isdisjoint(all_indices[1]))
        self.assertTrue(all_indices[0].isdisjoint(all_indices[2]))
        self.assertTrue(all_indices[1].isdisjoint(all_indices[2]))

    def test_predictions_preserve_order_restore_mode_and_integrate_with_metrics(self):
        import torch

        network = build_network()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network[0].weight[0, 0] = 1.0
            network[2].weight[0, 0] = 1.0
            network[4].weight[0, 0] = 200.0
            network[4].bias[0] = -100.0
        view = DatasetView(
            indices=(21, 22),
            features=((-1.0,) + (0.0,) * 17, (1.0,) + (0.0,) * 17),
            labels=(0, 1),
        )
        network.train()
        prediction = predict_view(network, view)
        self.assertTrue(network.training)
        self.assertEqual(prediction.indices, view.indices)
        self.assertEqual(prediction.labels, view.labels)
        self.assertEqual(prediction.logits, (-100.0, 100.0))
        self.assertEqual(prediction.scores, (0.0, 1.0))
        for score, logit in zip(prediction.scores, prediction.logits):
            expected = 1.0 / (1.0 + math.exp(-logit))
            self.assertAlmostEqual(score, expected, places=6)

        result = evaluate_scores(
            prediction.labels, prediction.scores, logits=prediction.logits)
        self.assertEqual(result["predicted_labels"], [0, 1])
        self.assertTrue(math.isfinite(result["values"]["bce_loss"]))
        self.assertIsNone(result["undefined_reasons"]["bce_loss"])

        singleton_records = [
            ((prediction.indices[offset],), (prediction.labels[offset],),
             (prediction.scores[offset],))
            for offset in range(2)
        ]
        global_result = evaluate_global(
            singleton_records, prediction.indices,
            logits=[(value,) for value in prediction.logits],
        )
        self.assertEqual(global_result, result)

    def test_train_blocks_rejects_invalid_inputs_and_default_seed_gate(self):
        valid = _view(8)
        network = build_network()
        state = initial_state_dict(42)
        invalid_calls = (
            lambda: train_blocks(network, "bad", model_family="mlp_centralized",
                                 training_seed=7, initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, _view(8, width=17),
                                 model_family="mlp_centralized", training_seed=7,
                                 initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, _view(2, labels=(0, 2)),
                                 model_family="mlp_centralized", training_seed=7,
                                 initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="mlp_centralized",
                                 client_id="client-0", training_seed=7,
                                 initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="mlp_local",
                                 training_seed=7, initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="other", training_seed=7,
                                 initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="mlp_centralized",
                                 training_seed=True, initial_state=state,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="mlp_centralized",
                                 training_seed=7, initial_state=state, blocks=0,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="mlp_centralized",
                                 training_seed=7, initial_state=state, learning_rate=0,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
            lambda: train_blocks(network, valid, model_family="mlp_centralized",
                                 training_seed=7, initial_state=None,
                                 shuffle_seed_fn=_synthetic_shuffle_seed),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(MLPError):
                call()

        with self.assertRaises(SeedingError):
            train_blocks(
                build_network(), valid, model_family="mlp_centralized",
                training_seed=7, blocks=1, epochs_per_block=1,
                initial_state=initial_state_dict(42),
            )


if __name__ == "__main__":
    unittest.main()
