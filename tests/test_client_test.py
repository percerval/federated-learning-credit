"""Testes da derivação ``client_test`` sem executar qualquer treinamento."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import unittest
from unittest import mock

import research.benchmark as benchmark_api
import research.benchmark.client_test as client_test_module
from research.benchmark.client_test import (
    ClientTestError,
    derive_client_tests,
)
from research.benchmark.data import DatasetView, load_experimental_data


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "hmeq.csv"
DATA_RECORD = ROOT / "results/data/benchmark-v1-split.json"
EXPECTED_CLIENT_IDS = ("client-0", "client-1", "client-2")


def _view(records):
    return DatasetView(
        indices=tuple(record[0] for record in records),
        features=tuple((float(record[0]), float(record[0] + 1)) for record in records),
        labels=tuple(record[1] for record in records),
    )


def _independent_expected(records):
    assigned = {client_id: [] for client_id in EXPECTED_CLIENT_IDS}
    by_index = {record[0]: record for record in records}
    for label in (0, 1):
        def key(index):
            payload = {
                "execution_spec_id": "benchmark-v1-execution-1",
                "partition_seed": 44,
                "stream": "client_test_partition",
                "label": label,
                "index": index,
            }
            canonical = json.dumps(
                payload, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(canonical).digest(), index

        indices = sorted(
            (index for index, record in by_index.items() if record[1] == label),
            key=key,
        )
        for position, index in enumerate(indices):
            assigned[EXPECTED_CLIENT_IDS[position % 3]].append(index)
    return {
        client_id: tuple(sorted(indices))
        for client_id, indices in assigned.items()
    }


class ClientTestSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.records = (
            (91, 1), (7, 0), (42, 1), (18, 0), (63, 1),
            (5, 0), (74, 1), (29, 0), (88, 0), (31, 1),
        )

    def test_matches_independent_canonical_sha256_derivation(self):
        view = _view(self.records)
        result = derive_client_tests(view)
        expected = _independent_expected(self.records)

        self.assertIsInstance(result, MappingProxyType)
        self.assertEqual(tuple(result), EXPECTED_CLIENT_IDS)
        self.assertEqual(
            {client_id: result[client_id].indices
             for client_id in EXPECTED_CLIENT_IDS},
            expected,
        )
        self.assertIs(benchmark_api.derive_client_tests, derive_client_tests)
        self.assertIs(benchmark_api.ClientTestError, ClientTestError)
        self.assertIn("derive_client_tests", benchmark_api.__all__)
        self.assertIn("ClientTestError", benchmark_api.__all__)

    def test_uses_index_as_tie_break_when_digests_match(self):
        records = ((9, 0), (1, 0), (5, 0), (3, 0))
        with mock.patch.object(
            client_test_module, "_partition_digest", return_value=b"same-digest",
        ):
            result = derive_client_tests(_view(records))

        self.assertEqual(result["client-0"].indices, (1, 9))
        self.assertEqual(result["client-1"].indices, (3,))
        self.assertEqual(result["client-2"].indices, (5,))

    def test_rejects_nested_mutable_feature_values(self):
        mutable_value = []
        view = DatasetView(
            indices=(1,), features=((mutable_value,),), labels=(0,),
        )

        with self.assertRaisesRegex(ClientTestError, "não numérica"):
            derive_client_tests(view)

    def test_is_deterministic_and_independent_of_input_order(self):
        forward = derive_client_tests(_view(self.records))
        reverse = derive_client_tests(_view(tuple(reversed(self.records))))
        for client_id in EXPECTED_CLIENT_IDS:
            self.assertEqual(forward[client_id], reverse[client_id])
            self.assertEqual(forward[client_id].indices,
                             tuple(sorted(forward[client_id].indices)))

    def test_preserves_records_with_exact_coverage_and_disjointness(self):
        source = _view(self.records)
        result = derive_client_tests(source)
        source_by_index = {
            index: (features, label)
            for index, features, label in zip(
                source.indices, source.features, source.labels,
            )
        }
        covered = set()
        for client_id in EXPECTED_CLIENT_IDS:
            view = result[client_id]
            self.assertFalse(covered.intersection(view.indices))
            covered.update(view.indices)
            for index, features, label in zip(view.indices, view.features, view.labels):
                self.assertEqual((features, label), source_by_index[index])
        self.assertEqual(covered, set(source.indices))
        self.assertEqual(sum(map(len, result.values())), len(source))

        for label in (0, 1):
            counts = [Counter(result[client_id].labels)[label]
                      for client_id in EXPECTED_CLIENT_IDS]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_rejects_invalid_inputs(self):
        invalid = (
            None,
            DatasetView(indices=(), features=(), labels=()),
            DatasetView(indices=(1, 1), features=((0.0,), (1.0,)), labels=(0, 1)),
            DatasetView(indices=(-1,), features=((0.0,),), labels=(0,)),
            DatasetView(indices=(True,), features=((0.0,),), labels=(0,)),
            DatasetView(indices=(1,), features=((0.0,),), labels=(2,)),
            DatasetView(indices=(1,), features=((0.0,),), labels=(True,)),
            DatasetView(indices=(1,), features=([0.0],), labels=(0,)),
            DatasetView(indices=(1,), features=((float("nan"),),), labels=(0,)),
            DatasetView(indices=(1,), features=((float("inf"),),), labels=(0,)),
            DatasetView(indices=(1,), features=((),), labels=(0,)),
            DatasetView(
                indices=(1, 2), features=((0.0,), (1.0, 2.0)), labels=(0, 1),
            ),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ClientTestError):
                derive_client_tests(value)


class ClientTestOfficialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_experimental_data(DATASET, DATA_RECORD)
        cls.result = derive_client_tests(cls.data.test)

    def test_is_compatible_with_frozen_e2_test_partition(self):
        covered = set()
        global_counts = Counter(self.data.test.labels)
        for position, client_id in enumerate(EXPECTED_CLIENT_IDS):
            view = self.result[client_id]
            self.assertEqual(view.indices, tuple(sorted(view.indices)))
            self.assertFalse(covered.intersection(view.indices))
            covered.update(view.indices)
            expected_count = sum(
                (count + 2 - position) // 3
                for count in global_counts.values()
            )
            self.assertEqual(len(view), expected_count)

        self.assertEqual(covered, set(self.data.test.indices))
        self.assertEqual(sum(map(len, self.result.values())), len(self.data.test))
        self.assertFalse(covered.intersection(self.data.train.indices))
        self.assertFalse(covered.intersection(self.data.validation.indices))


if __name__ == "__main__":
    unittest.main()
