"""Testes sintéticos do monitoramento MLP por bloco, sem benchmark oficial."""

from dataclasses import FrozenInstanceError
import importlib.util
import inspect
import unittest
from unittest import mock

import research.benchmark as benchmark_api
from research.benchmark import mlp as mlp_module
from research.benchmark.data import DatasetView
from research.benchmark.mlp import (
    MonitoredTrainingReport,
    build_network,
    clone_state_dict,
    final_state,
    initial_state_dict,
    train_blocks,
    train_blocks_monitored,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


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


def _shuffle_seed(training_seed, namespace, *, client_id, step, epoch):
    namespace_offset = {"central": 0, "local": 100}[namespace]
    client_offset = 0 if client_id is None else int(client_id[-1]) + 1
    return training_seed * 10_000 + namespace_offset + client_offset * 1000 \
        + step * 10 + epoch


@unittest.skipUnless(HAS_TORCH, "PyTorch indisponível")
class MLPMonitoringTests(unittest.TestCase):
    def test_final_parameters_exactly_match_original_for_central_and_local(self):
        import torch

        cases = (
            ("mlp_centralized", None, _view(21, start=10_000)),
            ("mlp_local", "client-1", _view(21, start=20_000, shift=0.4)),
        )
        validation = _view(8, start=40_000, shift=0.2)
        for family, client_id, train in cases:
            with self.subTest(family=family, client_id=client_id):
                initial = initial_state_dict(42)
                snapshot = clone_state_dict(initial)
                original_network = build_network()
                original_report = train_blocks(
                    original_network, train, model_family=family,
                    client_id=client_id, training_seed=7,
                    initial_state=initial, blocks=3, epochs_per_block=2,
                    batch_size=8, learning_rate=0.02,
                    shuffle_seed_fn=_shuffle_seed,
                )
                monitored_network = build_network()
                monitored_report = train_blocks_monitored(
                    monitored_network, train, validation,
                    model_family=family, client_id=client_id,
                    training_seed=7, initial_state=initial,
                    blocks=3, epochs_per_block=2, batch_size=8,
                    learning_rate=0.02, shuffle_seed_fn=_shuffle_seed,
                )

                original_state = final_state(original_network)
                monitored_state = final_state(monitored_network)
                self.assertTrue(all(
                    torch.equal(original_state[key], monitored_state[key])
                    for key in original_state
                ))
                self.assertEqual(monitored_report.training, original_report)
                self.assertTrue(all(
                    torch.equal(initial[key], snapshot[key]) for key in initial
                ))

    def test_metrics_are_deterministic_ordered_and_deeply_immutable(self):
        train = _view(19, start=10_000)
        validation = _view(9, start=40_000, shift=0.2)
        initial = initial_state_dict(42)

        reports = []
        for _ in range(2):
            reports.append(train_blocks_monitored(
                build_network(), train, validation,
                model_family="mlp_centralized", training_seed=7,
                initial_state=initial, blocks=3, epochs_per_block=1,
                batch_size=8, shuffle_seed_fn=_shuffle_seed,
            ))

        first, second = reports
        self.assertIsInstance(first, MonitoredTrainingReport)
        self.assertEqual(first, second)
        self.assertEqual(tuple(block.block_index for block in first.blocks), (0, 1, 2))
        self.assertEqual(len(first.blocks), first.training.blocks)
        for position, block in enumerate(first.blocks):
            self.assertEqual(block.training.step, position)
            self.assertEqual(block.validation_indices, validation.indices)
            self.assertEqual(block.validation_metrics["sample_count"], len(validation))
            self.assertEqual(set(block.validation_metrics["values"]), {
                "bce_loss", "accuracy", "precision", "recall", "f1",
                "average_precision", "roc_auc",
            })
        with self.assertRaises(FrozenInstanceError):
            first.blocks[0].block_index = 9
        with self.assertRaises(TypeError):
            first.blocks[0].validation_metrics["sample_count"] = 0
        with self.assertRaises(TypeError):
            first.blocks[0].validation_metrics["values"]["accuracy"] = 0.0

    def test_validation_is_observational_and_api_has_no_test_input(self):
        import torch

        train = _view(17, start=10_000)
        validation = _view(7, start=40_000)
        observed_views = []
        events = []
        real_predict = mlp_module.predict_view
        real_train_step = mlp_module.train_step

        def train_spy(network, train_view, **kwargs):
            step_report = real_train_step(network, train_view, **kwargs)
            events.append((
                "train", kwargs["step"], final_state(network), network.training,
            ))
            return step_report

        def predict_spy(network, view):
            observed_views.append(view)
            before = final_state(network)
            mode_before = network.training
            prediction = real_predict(network, view)
            events.append((
                "validation", len(observed_views) - 1, before,
                final_state(network), mode_before, network.training,
            ))
            return prediction

        self.assertNotIn("test", inspect.signature(train_blocks_monitored).parameters)
        torch.manual_seed(903)
        rng_before = torch.random.get_rng_state().clone()
        with (
            mock.patch.object(mlp_module, "train_step", side_effect=train_spy),
            mock.patch.object(mlp_module, "predict_view", side_effect=predict_spy),
        ):
            report = train_blocks_monitored(
                build_network(), train, validation,
                model_family="mlp_centralized", training_seed=7,
                initial_state=initial_state_dict(42), blocks=3,
                epochs_per_block=1, batch_size=8,
                shuffle_seed_fn=_shuffle_seed,
            )

        self.assertEqual(observed_views, [validation, validation, validation])
        self.assertEqual(
            tuple((event[0], event[1]) for event in events),
            (("train", 0), ("validation", 0),
             ("train", 1), ("validation", 1),
             ("train", 2), ("validation", 2)),
        )
        for position in range(0, len(events), 2):
            trained = events[position]
            validated = events[position + 1]
            self.assertTrue(all(
                torch.equal(trained[2][key], validated[2][key])
                for key in trained[2]
            ))
            self.assertTrue(all(
                torch.equal(validated[2][key], validated[3][key])
                for key in validated[2]
            ))
            self.assertTrue(trained[3])
            self.assertTrue(validated[4])
            self.assertTrue(validated[5])
        self.assertEqual(len(report.blocks), 3)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertIs(benchmark_api.train_blocks_monitored, train_blocks_monitored)
        self.assertIn("train_blocks_monitored", benchmark_api.__all__)

    def test_rejects_invalid_validation_scope_before_first_adam(self):
        train = _view(8, start=10_000)
        valid = _view(3, start=40_000)
        invalid_views = (
            DatasetView(
                indices=tuple(reversed(valid.indices)),
                features=tuple(reversed(valid.features)),
                labels=tuple(reversed(valid.labels)),
            ),
            DatasetView(
                indices=(40_000, 40_000),
                features=valid.features[:2], labels=valid.labels[:2],
            ),
            DatasetView(
                indices=(train.indices[0],),
                features=(train.features[0],), labels=(train.labels[0],),
            ),
        )

        for validation in invalid_views:
            with self.subTest(indices=validation.indices):
                with mock.patch("torch.optim.Adam") as adam:
                    with self.assertRaises(mlp_module.MLPError):
                        train_blocks_monitored(
                            build_network(), train, validation,
                            model_family="mlp_centralized", training_seed=7,
                            initial_state=initial_state_dict(42), blocks=1,
                            epochs_per_block=1, batch_size=8,
                            shuffle_seed_fn=_shuffle_seed,
                        )
                adam.assert_not_called()


if __name__ == "__main__":
    unittest.main()
