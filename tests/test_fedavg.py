"""Testes sintéticos da agregação e coordenação FedAvg, sem benchmark oficial."""

from dataclasses import FrozenInstanceError
import importlib.util
import unittest
from unittest import mock

import research.benchmark as benchmark_api
from research.benchmark import fedavg as fedavg_module
from research.benchmark.data import DatasetView
from research.benchmark.fedavg import (
    NO_SERIALIZATION_NOTE,
    FedAvgError,
    FedAvgTrainingReport,
    aggregate_state_dicts,
    train_fedavg,
)
from research.benchmark.mlp import (
    MLPError,
    build_network,
    clone_state_dict,
    final_state,
    initial_state_dict,
)
from research.benchmark.seeding import paired_initialization_seed


HAS_TORCH = importlib.util.find_spec("torch") is not None
CLIENTS = ("client-0", "client-1", "client-2")


def _view(count, *, start, shift=0.0):
    return DatasetView(
        indices=tuple(start + index for index in range(count)),
        features=tuple(
            tuple((((index + column) % 7) - 3) / 3.0 + shift
                  for column in range(18))
            for index in range(count)
        ),
        labels=tuple(index % 2 for index in range(count)),
    )


def _clients():
    return {
        "client-0": _view(5, start=10_000, shift=0.0),
        "client-1": _view(7, start=20_000, shift=0.3),
        "client-2": _view(9, start=30_000, shift=0.6),
    }


def _synthetic_shuffle_seed(training_seed, namespace, *, client_id, step, epoch):
    client_offset = int(client_id[-1]) + 1
    namespace_offset = {"federated": 100}[namespace]
    return training_seed * 100_000 + namespace_offset + client_offset * 1000 + step * 10 + epoch


@unittest.skipUnless(HAS_TORCH, "PyTorch indisponível")
class FedAvgAggregationTests(unittest.TestCase):
    @staticmethod
    def _states(values=(1.0, 3.0, 9.0)):
        import torch

        return {
            client_id: {
                "weight": torch.tensor([[value, value + 1]], dtype=torch.float32),
                "bias": torch.tensor([value - 1], dtype=torch.float32),
            }
            for client_id, value in zip(CLIENTS, values)
        }

    def test_weighted_average_uses_example_counts_not_simple_mean(self):
        import torch

        states = self._states()
        result = aggregate_state_dicts(
            states,
            {"client-0": 1, "client-1": 2, "client-2": 5},
        )

        expected_weight = torch.tensor([[6.5, 7.5]], dtype=torch.float32)
        expected_bias = torch.tensor([5.5], dtype=torch.float32)
        simple_mean = sum(state["weight"] for state in states.values()) / 3
        self.assertTrue(torch.equal(result["weight"], expected_weight))
        self.assertTrue(torch.equal(result["bias"], expected_bias))
        self.assertFalse(torch.equal(result["weight"], simple_mean))
        self.assertEqual(result["weight"].dtype, torch.float32)
        self.assertEqual(result["bias"].dtype, torch.float32)

    def test_result_is_deterministic_across_client_insertion_orders(self):
        import torch

        states = self._states()
        counts = {"client-0": 2, "client-1": 3, "client-2": 7}
        reversed_states = dict(reversed(tuple(states.items())))
        reversed_counts = dict(reversed(tuple(counts.items())))

        first = aggregate_state_dicts(states, counts)
        second = aggregate_state_dicts(reversed_states, reversed_counts)
        self.assertEqual(tuple(first), tuple(second))
        self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))

    def test_inputs_are_not_mutated_and_result_has_no_aliases(self):
        import torch

        states = self._states()
        snapshots = {
            client_id: {name: tensor.clone() for name, tensor in state.items()}
            for client_id, state in states.items()
        }
        torch.manual_seed(731)
        rng_before = torch.random.get_rng_state().clone()
        result = aggregate_state_dicts(
            states,
            {"client-0": 1, "client-1": 1, "client-2": 1},
        )

        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for client_id in CLIENTS:
            for name in result:
                self.assertTrue(torch.equal(states[client_id][name],
                                            snapshots[client_id][name]))
                self.assertNotEqual(result[name].data_ptr(),
                                    states[client_id][name].data_ptr())
        result["weight"].add_(100)
        self.assertTrue(all(
            torch.equal(states[client_id]["weight"], snapshots[client_id]["weight"])
            for client_id in CLIENTS
        ))

    def test_rejects_missing_and_extra_clients(self):
        states = self._states()
        counts = {client_id: 1 for client_id in CLIENTS}
        invalid = (
            ({key: value for key, value in states.items() if key != "client-2"}, counts),
            ({**states, "client-3": states["client-0"]}, counts),
            (states, {"client-0": 1, "client-1": 1}),
            (states, {**counts, "client-3": 1}),
        )
        for local_states, example_counts in invalid:
            with self.subTest(clients=(tuple(local_states), tuple(example_counts))):
                with self.assertRaises(FedAvgError):
                    aggregate_state_dicts(local_states, example_counts)

    def test_rejects_invalid_counts(self):
        states = self._states()
        for invalid_count in (0, -1, 1.5, True, "1"):
            counts = {client_id: 1 for client_id in CLIENTS}
            counts["client-1"] = invalid_count
            with self.subTest(count=invalid_count), self.assertRaises(FedAvgError):
                aggregate_state_dicts(states, counts)

    def test_rejects_invalid_state_structure_dtype_and_values(self):
        import torch

        counts = {client_id: 1 for client_id in CLIENTS}
        invalid_states = []

        missing_key = self._states()
        missing_key["client-1"] = {"weight": missing_key["client-1"]["weight"]}
        invalid_states.append(missing_key)

        extra_key = self._states()
        extra_key["client-1"]["extra"] = torch.tensor([1.0])
        invalid_states.append(extra_key)

        wrong_shape = self._states()
        wrong_shape["client-1"]["weight"] = torch.ones(3, dtype=torch.float32)
        invalid_states.append(wrong_shape)

        wrong_dtype = self._states()
        wrong_dtype["client-1"]["weight"] = torch.ones(1, 2, dtype=torch.float64)
        invalid_states.append(wrong_dtype)

        not_tensor = self._states()
        not_tensor["client-1"]["weight"] = [[1.0, 2.0]]
        invalid_states.append(not_tensor)

        for invalid_value in (float("nan"), float("inf"), float("-inf")):
            nonfinite = self._states()
            nonfinite["client-1"]["bias"] = torch.tensor(
                [invalid_value], dtype=torch.float32,
            )
            invalid_states.append(nonfinite)

        for position, states in enumerate(invalid_states):
            with self.subTest(position=position), self.assertRaises(FedAvgError):
                aggregate_state_dicts(states, counts)


@unittest.skipUnless(HAS_TORCH, "PyTorch indisponível")
class FedAvgCoordinatorTests(unittest.TestCase):
    def test_public_api_exports_only_required_federated_surface(self):
        expected = {
            "FedAvgError",
            "FedAvgTrainingReport",
            "FederatedClientRoundReport",
            "FederatedRoundReport",
            "aggregate_state_dicts",
            "train_fedavg",
        }
        self.assertTrue(expected.issubset(benchmark_api.__all__))
        self.assertNotIn("NO_SERIALIZATION_NOTE", benchmark_api.__all__)
        self.assertNotIn("train_step", benchmark_api.__all__)
        self.assertIs(benchmark_api.train_fedavg, train_fedavg)
        self.assertIs(benchmark_api.aggregate_state_dicts, aggregate_state_dicts)

    def test_one_short_round_reports_validation_costs_and_communication(self):
        import torch

        network = build_network()
        validation = _view(4, start=40_000)
        report = train_fedavg(
            network, _clients(), validation,
            training_seed=42, initial_state=initial_state_dict(42),
            rounds=1, epochs_per_round=1, batch_size=4,
            shuffle_seed_fn=_synthetic_shuffle_seed,
        )

        self.assertIsInstance(report, FedAvgTrainingReport)
        self.assertEqual(report.initialization_seed, paired_initialization_seed(42))
        self.assertEqual(report.client_ids, CLIENTS)
        self.assertEqual(len(report.round_reports), 1)
        round_report = report.round_reports[0]
        self.assertEqual(tuple(client.client_id for client in round_report.clients), CLIENTS)
        self.assertEqual(
            tuple(client.fit_indices for client in round_report.clients),
            tuple(_clients()[client_id].indices for client_id in CLIENTS),
        )
        self.assertEqual(round_report.validation_indices, validation.indices)
        self.assertEqual(round_report.validation_metrics["sample_count"], 4)
        self.assertEqual(round_report.examples_processed, 21)
        self.assertEqual(round_report.optimizer_steps, 7)
        self.assertEqual(round_report.examples_processed,
                         round_report.theoretical_examples)
        self.assertEqual(round_report.optimizer_steps,
                         round_report.theoretical_steps)
        self.assertEqual(report.examples_processed, 21)
        self.assertEqual(report.optimizer_steps, 7)
        self.assertEqual(report.examples_processed, report.theoretical_examples)
        self.assertEqual(report.optimizer_steps, report.theoretical_steps)
        parameter_bytes = 449 * torch.tensor([], dtype=torch.float32).element_size()
        self.assertEqual(round_report.logical_downloads, 3)
        self.assertEqual(round_report.logical_uploads, 3)
        self.assertEqual(round_report.theoretical_download_bytes, parameter_bytes * 3)
        self.assertEqual(round_report.theoretical_upload_bytes, parameter_bytes * 3)
        self.assertEqual(round_report.serialized_download_bytes, 0)
        self.assertEqual(round_report.serialized_upload_bytes, 0)
        self.assertEqual(round_report.serialization_note, NO_SERIALIZATION_NOTE)
        with self.assertRaises(FrozenInstanceError):
            report.rounds = 99
        with self.assertRaises(TypeError):
            round_report.validation_metrics["sample_count"] = 99

    def test_multiple_rounds_distribute_aggregate_and_use_independent_adam(self):
        import torch

        clients = _clients()
        validation = _view(4, start=40_000)
        initial_state = initial_state_dict(42)
        starts = []
        aggregates = []
        seed_calls = []
        validation_calls = []
        optimizers = []
        real_train_step = fedavg_module.train_step
        real_aggregate = fedavg_module.aggregate_state_dicts
        real_predict = fedavg_module.predict_view
        real_adam = torch.optim.Adam

        def step_spy(network, train_view, **kwargs):
            starts.append((
                kwargs["step"], kwargs["client_id"], final_state(network),
            ))
            return real_train_step(network, train_view, **kwargs)

        def aggregate_spy(states, counts):
            result = real_aggregate(states, counts)
            aggregates.append(clone_state_dict(result))
            return result

        def predict_spy(network, view):
            validation_calls.append(view)
            return real_predict(network, view)

        def adam_spy(*args, **kwargs):
            optimizer = real_adam(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        def seed_spy(training_seed, namespace, *, client_id, step, epoch):
            seed_calls.append((training_seed, namespace, client_id, step, epoch))
            return _synthetic_shuffle_seed(
                training_seed, namespace, client_id=client_id,
                step=step, epoch=epoch,
            )

        with (
            mock.patch.object(fedavg_module, "train_step", side_effect=step_spy),
            mock.patch.object(fedavg_module, "aggregate_state_dicts",
                              side_effect=aggregate_spy),
            mock.patch.object(fedavg_module, "predict_view", side_effect=predict_spy),
            mock.patch("torch.optim.Adam", side_effect=adam_spy),
        ):
            report = train_fedavg(
                build_network(), clients, validation,
                training_seed=7, initial_state=initial_state,
                rounds=2, epochs_per_round=2, batch_size=4,
                shuffle_seed_fn=seed_spy,
            )

        self.assertEqual(len(starts), 6)
        for _, _, state in starts[:3]:
            self.assertTrue(all(torch.equal(state[key], initial_state[key])
                                for key in state))
        for _, _, state in starts[3:]:
            self.assertTrue(all(torch.equal(state[key], aggregates[0][key])
                                for key in state))
        for group in (starts[:3], starts[3:]):
            first_state = group[0][2]
            self.assertTrue(all(
                torch.equal(first_state[key], item[2][key])
                for item in group[1:] for key in first_state
            ))
        self.assertEqual(len(aggregates), 2)
        self.assertEqual(len(optimizers), 6)
        self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 6)
        self.assertEqual(len({id(optimizer.state) for optimizer in optimizers}), 6)
        self.assertEqual(validation_calls, [validation, validation])
        self.assertEqual(seed_calls, [
            (7, "federated", client_id, round_index, epoch)
            for round_index in range(2)
            for client_id in CLIENTS
            for epoch in range(2)
        ])
        self.assertEqual(
            tuple(client.shuffle_seeds
                  for round_report in report.round_reports
                  for client in round_report.clients),
            tuple(tuple(_synthetic_shuffle_seed(
                7, "federated", client_id=client_id,
                step=round_index, epoch=epoch,
            ) for epoch in range(2))
                  for round_index in range(2) for client_id in CLIENTS),
        )

    def test_same_inputs_produce_identical_state_and_report_without_rng_consumption(self):
        import torch

        clients = _clients()
        validation = _view(4, start=40_000)
        initial_state = initial_state_dict(42)
        torch.manual_seed(903)
        rng_before = torch.random.get_rng_state().clone()

        first_network = build_network()
        first_report = train_fedavg(
            first_network, clients, validation,
            training_seed=7, initial_state=initial_state,
            rounds=2, epochs_per_round=1, batch_size=4,
            shuffle_seed_fn=_synthetic_shuffle_seed,
        )
        second_network = build_network()
        second_report = train_fedavg(
            second_network, clients, validation,
            training_seed=7, initial_state=initial_state,
            rounds=2, epochs_per_round=1, batch_size=4,
            shuffle_seed_fn=_synthetic_shuffle_seed,
        )

        first_state = final_state(first_network)
        second_state = final_state(second_network)
        self.assertTrue(all(torch.equal(first_state[key], second_state[key])
                            for key in first_state))
        self.assertEqual(first_report, second_report)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))

    def test_requires_all_clients_and_disjoint_train_validation_indices(self):
        clients = _clients()
        validation = _view(4, start=40_000)
        invalid_clients = {key: value for key, value in clients.items()
                           if key != "client-2"}
        with self.assertRaises(FedAvgError):
            train_fedavg(
                build_network(), invalid_clients, validation,
                training_seed=7, initial_state=initial_state_dict(42),
                rounds=1, epochs_per_round=1, batch_size=4,
                shuffle_seed_fn=_synthetic_shuffle_seed,
            )

        unsorted_validation = DatasetView(
            indices=(40_001, 40_000),
            features=validation.features[:2],
            labels=validation.labels[:2],
        )
        with self.assertRaisesRegex(FedAvgError, "ordem crescente"):
            train_fedavg(
                build_network(), clients, unsorted_validation,
                training_seed=7, initial_state=initial_state_dict(42),
                rounds=1, epochs_per_round=1, batch_size=4,
                shuffle_seed_fn=_synthetic_shuffle_seed,
            )

        overlapping_validation = DatasetView(
            indices=(clients["client-0"].indices[0],),
            features=(clients["client-0"].features[0],),
            labels=(clients["client-0"].labels[0],),
        )
        with self.assertRaises(FedAvgError):
            train_fedavg(
                build_network(), clients, overlapping_validation,
                training_seed=7, initial_state=initial_state_dict(42),
                rounds=1, epochs_per_round=1, batch_size=4,
                shuffle_seed_fn=_synthetic_shuffle_seed,
            )

    def test_rejects_unpaired_official_initial_state_before_any_adam(self):
        with mock.patch("torch.optim.Adam") as adam:
            with self.assertRaisesRegex(MLPError, "não corresponde.*training_seed"):
                train_fedavg(
                    build_network(), _clients(), _view(4, start=40_000),
                    training_seed=43, initial_state=initial_state_dict(42),
                    rounds=1, epochs_per_round=1, batch_size=4,
                    shuffle_seed_fn=_synthetic_shuffle_seed,
                )
        adam.assert_not_called()
