"""
Unit tests for optimization.inference.generate_id_partitions.

generate_id_partitions splits an array of scan IDs into n_batch lists
using either 'round_robin' or 'block' assignment. Tests verify:
- All input IDs appear in the output exactly once
- The number of partitions equals n_batch
- round_robin and block produce different (but valid) orderings
- Edge cases: n_batch > len(id_array), single batch, etc.
"""

import numpy as np
import pytest

from optimization.inference import generate_id_partitions


def _all_ids(partitions):
    """Flatten all partition lists into a sorted list."""
    return sorted(x for part in partitions for x in part)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestGenerateIdPartitions:
    def test_returns_list_of_lists(self):
        result = generate_id_partitions(np.arange(10), n_batch=3, how="round_robin")
        assert isinstance(result, list)
        assert all(isinstance(p, list) for p in result)

    def test_number_of_partitions_equals_n_batch(self):
        for n in [1, 2, 5, 10]:
            result = generate_id_partitions(np.arange(50), n_batch=n, how="round_robin")
            assert len(result) == n

    def test_all_ids_present_round_robin(self):
        ids = np.arange(100)
        result = generate_id_partitions(ids, n_batch=4, how="round_robin")
        assert _all_ids(result) == sorted(ids.tolist())

    def test_all_ids_present_block(self):
        ids = np.arange(100)
        result = generate_id_partitions(ids, n_batch=4, how="block")
        assert _all_ids(result) == sorted(ids.tolist())

    def test_no_duplicate_ids_round_robin(self):
        ids = np.arange(60)
        result = generate_id_partitions(ids, n_batch=3, how="round_robin")
        flat = [x for part in result for x in part]
        assert len(flat) == len(set(flat))

    def test_no_duplicate_ids_block(self):
        # n_edge_counts defaults to 50; need total >> 2*n_edge_counts for
        # block sizes to be positive and non-overlapping
        ids = np.arange(600)
        result = generate_id_partitions(ids, n_batch=3, how="block")
        flat = [x for part in result for x in part]
        assert len(flat) == len(set(flat))

    def test_single_batch_contains_all(self):
        ids = np.arange(30)
        result = generate_id_partitions(ids, n_batch=1, how="round_robin")
        assert len(result) == 1
        assert _all_ids(result) == sorted(ids.tolist())

    def test_round_robin_distributes_evenly(self):
        ids = np.arange(12)
        result = generate_id_partitions(ids, n_batch=4, how="round_robin")
        sizes = [len(p) for p in result]
        assert max(sizes) - min(sizes) <= 1  # balanced within 1

    def test_round_robin_vs_block_differ(self):
        ids = np.arange(100)
        rr = generate_id_partitions(ids, n_batch=5, how="round_robin")
        bl = generate_id_partitions(ids, n_batch=5, how="block")
        # Content is the same overall but partition assignment differs
        assert _all_ids(rr) == _all_ids(bl)
        # At least one partition should differ in membership
        assert any(sorted(rr[i]) != sorted(bl[i]) for i in range(5))

    def test_non_contiguous_ids(self):
        """IDs don't have to be 0..N-1."""
        ids = np.array([10, 20, 30, 40, 50, 60])
        result = generate_id_partitions(ids, n_batch=3, how="round_robin")
        assert _all_ids(result) == sorted(ids.tolist())

    def test_large_n_batch_relative_to_ids(self):
        """More batches than IDs: some partitions will be empty."""
        ids = np.arange(5)
        result = generate_id_partitions(ids, n_batch=10, how="round_robin")
        assert len(result) == 10
        assert _all_ids(result) == sorted(ids.tolist())

    # -----------------------------------------------------------------------
    # rt_chunk mode
    # -----------------------------------------------------------------------

    def test_rt_chunk_round_robin_all_ids_present(self):
        ids = np.arange(100)
        result = generate_id_partitions(ids, n_batch=4, how="round_robin", rt_chunk=10)
        assert _all_ids(result) == sorted(ids.tolist())

    def test_rt_chunk_block_all_ids_present(self):
        # With rt_chunk=10 and 1000 IDs there are 100 RT chunks >> n_edge_counts=50
        ids = np.arange(1000)
        result = generate_id_partitions(ids, n_batch=4, how="block", rt_chunk=10)
        assert _all_ids(result) == sorted(ids.tolist())

    def test_rt_chunk_groups_consecutive_scans_together(self):
        """Scans in the same RT chunk should end up in the same partition (round_robin)."""
        ids = np.arange(20)  # chunks of size 5: [0-4], [5-9], [10-14], [15-19]
        result = generate_id_partitions(ids, n_batch=4, how="round_robin", rt_chunk=5)
        # Each chunk should not be split across partitions
        for chunk_start in range(0, 20, 5):
            chunk_ids = set(range(chunk_start, chunk_start + 5))
            assigned_partitions = {
                i for i, part in enumerate(result) if chunk_ids & set(part)
            }
            assert len(assigned_partitions) == 1, (
                f"Chunk {chunk_start}-{chunk_start+4} was split across partitions"
            )
