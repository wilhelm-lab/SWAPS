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
import pandas as pd
import pytest

from optimization.inference import (
    generate_id_partitions,
    collapse_candidates_by_confounder_group,
    _prepare_sparse_matrices,
    sparse_encode_divide_and_conquer_with_residual_stats,
)
from postprocessing.helper import expand_group_ids_to_members


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


# ---------------------------------------------------------------------------
# collapse_candidates_by_confounder_group (coSWA confounder merging)
# ---------------------------------------------------------------------------


def _candidate_row(
    mz_rank, iso_mz, iso_ab, rt_left=1.0, rt_right=2.0, group_id=-1, group_iso_mz=None, group_iso_ab=None
):
    iso_mz = np.asarray(iso_mz, dtype=float)
    iso_ab = np.asarray(iso_ab, dtype=float)
    return {
        "mz_rank": mz_rank,
        "IsoMZ": iso_mz,
        "IsoAbundance": iso_ab,
        "mz_length": len(iso_mz),
        "RT_search_left": rt_left,
        "RT_search_right": rt_right,
        "confounder_group_id": group_id,
        "GroupIsoMZ": np.asarray(group_iso_mz, dtype=float) if group_iso_mz is not None else iso_mz,
        "GroupIsoAbundance": np.asarray(group_iso_ab, dtype=float) if group_iso_ab is not None else iso_ab,
        "GroupMzLength": len(group_iso_mz) if group_iso_mz is not None else len(iso_mz),
        "GroupRT_search_left": rt_left,
        "GroupRT_search_right": rt_right,
    }


class TestCollapseCandidatesByConfounderGroup:
    def test_no_group_column_passthrough(self):
        df = pd.DataFrame([_candidate_row(1, [500.0], [1.0])]).drop(
            columns=["confounder_group_id"]
        )
        result = collapse_candidates_by_confounder_group(df)
        pd.testing.assert_frame_equal(result, df)

    def test_all_solo_passthrough(self):
        df = pd.DataFrame(
            [_candidate_row(1, [500.0], [1.0]), _candidate_row(2, [600.0], [1.0])]
        )
        result = collapse_candidates_by_confounder_group(df)
        assert sorted(result["mz_rank"]) == [1, 2]

    def test_group_collapses_to_one_representative_row(self):
        df = pd.DataFrame(
            [
                _candidate_row(
                    1, [500.0, 500.5], [0.7, 0.3], group_id=1001,
                    group_iso_mz=[500.0, 500.5], group_iso_ab=[0.667, 0.333],
                ),
                _candidate_row(
                    2, [500.0, 500.5], [0.65, 0.35], group_id=1001,
                    group_iso_mz=[500.0, 500.5], group_iso_ab=[0.667, 0.333],
                ),
                _candidate_row(3, [600.0], [1.0]),  # solo
            ]
        )
        result = collapse_candidates_by_confounder_group(df)
        # solo candidate untouched, group collapsed to a single row keyed by group id
        assert sorted(result["mz_rank"]) == [3, 1001]
        rep = result.loc[result["mz_rank"] == 1001].iloc[0]
        assert np.array_equal(rep["IsoMZ"], np.array([500.0, 500.5]))
        assert np.array_equal(rep["IsoAbundance"], np.array([0.667, 0.333]))
        assert rep["mz_length"] == 2

    def test_solo_candidate_rows_are_untouched(self):
        df = pd.DataFrame(
            [
                _candidate_row(1, [500.0, 500.5], [0.7, 0.3], group_id=1001),
                _candidate_row(2, [500.0, 500.5], [0.65, 0.35], group_id=1001),
                _candidate_row(3, [600.0, 600.5], [0.8, 0.2]),  # solo
            ]
        )
        result = collapse_candidates_by_confounder_group(df)
        solo_row = result.loc[result["mz_rank"] == 3].iloc[0]
        original_row = df.loc[df["mz_rank"] == 3].iloc[0]
        assert np.array_equal(solo_row["IsoMZ"], original_row["IsoMZ"])
        assert np.array_equal(solo_row["IsoAbundance"], original_row["IsoAbundance"])

    def test_multiple_independent_groups(self):
        df = pd.DataFrame(
            [
                _candidate_row(1, [500.0], [1.0], group_id=1001),
                _candidate_row(2, [500.0], [1.0], group_id=1001),
                _candidate_row(3, [700.0], [1.0], group_id=1003),
                _candidate_row(4, [700.0], [1.0], group_id=1003),
            ]
        )
        result = collapse_candidates_by_confounder_group(df)
        assert sorted(result["mz_rank"]) == [1001, 1003]


# ---------------------------------------------------------------------------
# Benchmark: merging confounders must not perturb solo candidates' SWA solve
# ---------------------------------------------------------------------------


# Two confounders A (mz_rank=1) and B (mz_rank=2) share near-identical
# isotope patterns at m/z ~500 (ambiguous, co-eluting MS1 signal at IM row 5,
# true injected intensity 700+300=1000 split across their two isotope peaks);
# C (mz_rank=3) is a solo candidate at a well-separated m/z (~505, orthogonal
# m/z bins and its own IM row 12) with an unambiguous, cleanly-matching
# signal (true injected intensity 800+200=1000).
_GROUP_ID = 1001
_IM_ROW_GROUP = 5
_IM_ROW_SOLO = 12
_TRUE_INTENSITY_GROUP = 700.0 + 300.0
_TRUE_INTENSITY_SOLO = 800.0 + 200.0


def _build_candidates(with_group: bool) -> pd.DataFrame:
    candidates = [
        _candidate_row(
            1, [500.0, 500.5], [0.7, 0.3], group_id=_GROUP_ID,
            group_iso_mz=[500.0, 500.5], group_iso_ab=[0.6667, 0.3333],
        ),
        _candidate_row(
            2, [500.0, 500.5], [0.65, 0.35], group_id=_GROUP_ID,
            group_iso_mz=[500.0, 500.5], group_iso_ab=[0.6667, 0.3333],
        ),
        _candidate_row(3, [505.0, 505.5], [0.8, 0.2]),  # solo, orthogonal m/z region
    ]
    df = pd.DataFrame(candidates)
    if with_group:
        df = collapse_candidates_by_confounder_group(df)
    return df.sort_values("mz_rank", ascending=True).reset_index(drop=True)


def _solve_synthetic_frame(df: pd.DataFrame):
    """Run the actual SWA candidate-array + sparse-encode pipeline on the
    given (already collapsed-or-not) candidate set. Returns
    (frame_act (im x n_candidates), all_id (sorted candidate row keys))."""
    mobility_values = pd.DataFrame({"mobility_values": np.linspace(0.5, 1.5, 20)})
    frame_data = pd.DataFrame(
        {
            "mz_values": [500.0, 500.5, 505.0, 505.5],
            "intensity_values": [700.0, 300.0, 800.0, 200.0],
            "mobility_values": [
                mobility_values["mobility_values"].iloc[_IM_ROW_GROUP],
                mobility_values["mobility_values"].iloc[_IM_ROW_GROUP],
                mobility_values["mobility_values"].iloc[_IM_ROW_SOLO],
                mobility_values["mobility_values"].iloc[_IM_ROW_SOLO],
            ],
        }
    )
    all_id = np.sort(df["mz_rank"].to_numpy())
    frame_array, candidate_array = _prepare_sparse_matrices(
        candidate_precursor_by_rt=df,
        frame_data=frame_data,
        all_id=all_id,
        ppm_tol=20,
        bin_width=0.01,
        use_ims=True,
        mobility_values=mobility_values,
    )
    frame_act = sparse_encode_divide_and_conquer_with_residual_stats(
        frame_array, candidate_array
    )
    return frame_act, all_id


class TestMergeConfoundersEndToEndSolve:
    """Exercises the real _prepare_sparse_matrices + sparse_encode pipeline
    with and without confounder merging, on a candidate set that has
    nonzero, physically meaningful signal both inside the confounder group
    (A, B) and outside it (solo candidate C)."""

    def test_solo_candidate_activation_identical_with_or_without_grouping(self):
        act_off, id_off = _solve_synthetic_frame(_build_candidates(with_group=False))
        act_on, id_on = _solve_synthetic_frame(_build_candidates(with_group=True))
        solo_off = act_off[:, int(np.where(id_off == 3)[0][0])]
        solo_on = act_on[:, int(np.where(id_on == 3)[0][0])]
        # C's m/z bins and IM row are orthogonal to the merged A/B group's,
        # so collapsing A+B into one candidate row must leave C's own
        # solved activation trace unchanged.
        np.testing.assert_allclose(solo_off, solo_on, rtol=1e-5, atol=1e-8)
        # sanity: the solo signal is not vacuously all-zero
        assert solo_off[_IM_ROW_SOLO] > 0
        assert solo_off.sum() == pytest.approx(solo_off[_IM_ROW_SOLO])

    def test_ungrouped_confounders_both_get_nonzero_competing_activation(self):
        """Without merging, A and B are separate, nearly-collinear rows in
        the candidate matrix: sklearn's sparse_encode('threshold', alpha=0)
        correlates the shared signal against each independently, so both
        end up with large, overlapping nonzero activation instead of the
        signal being cleanly attributed once -- this is exactly the
        competing/unstable behavior coSWA merging is meant to fix."""
        act_off, id_off = _solve_synthetic_frame(_build_candidates(with_group=False))
        a_col = int(np.where(id_off == 1)[0][0])
        b_col = int(np.where(id_off == 2)[0][0])
        act_a = act_off[_IM_ROW_GROUP, a_col]
        act_b = act_off[_IM_ROW_GROUP, b_col]
        assert act_a > 0
        assert act_b > 0
        # double-counting: each of A and B independently reconstructs close
        # to the full shared signal, so their sum overshoots the true total.
        assert act_a + act_b > _TRUE_INTENSITY_GROUP

    def test_grouped_confounders_collapse_to_one_stable_activation(self):
        """With merging, A and B are represented by a single candidate row
        (keyed by the group id), so the frame solve produces exactly one
        activation value for the whole group at the shared IM row -- no
        separate, competing per-candidate values exist to disagree."""
        act_on, id_on = _solve_synthetic_frame(_build_candidates(with_group=True))
        assert list(id_on) == [3, _GROUP_ID]  # group id sorts after real mz_ranks
        group_col = int(np.where(id_on == _GROUP_ID)[0][0])
        merged_act = act_on[_IM_ROW_GROUP, group_col]
        assert merged_act > 0

    def test_merging_reduces_double_counting_vs_the_naive_competing_sum(self):
        act_off, id_off = _solve_synthetic_frame(_build_candidates(with_group=False))
        act_on, id_on = _solve_synthetic_frame(_build_candidates(with_group=True))
        a_col = int(np.where(id_off == 1)[0][0])
        b_col = int(np.where(id_off == 2)[0][0])
        naive_competing_sum = (
            act_off[_IM_ROW_GROUP, a_col] + act_off[_IM_ROW_GROUP, b_col]
        )
        group_col = int(np.where(id_on == _GROUP_ID)[0][0])
        merged_act = act_on[_IM_ROW_GROUP, group_col]
        assert merged_act < naive_competing_sum

    def test_expanding_merged_activation_gives_members_identical_values(self):
        """After solving, expand_group_ids_to_members must duplicate the
        single merged activation value out to both member mz_ranks
        identically -- this is the "duplicate rows per member mz_rank"
        parquet output contract from the coSWA design."""
        act_on, id_on = _solve_synthetic_frame(_build_candidates(with_group=True))
        group_col = int(np.where(id_on == _GROUP_ID)[0][0])
        merged_act = float(act_on[_IM_ROW_GROUP, group_col])

        group_to_members = {_GROUP_ID: np.array([1, 2], dtype=np.uint32)}
        _, _, expanded_pept, expanded_data = expand_group_ids_to_members(
            [0], [_IM_ROW_GROUP], [_GROUP_ID], [merged_act], group_to_members
        )
        by_member = dict(zip(expanded_pept.tolist(), expanded_data.tolist()))
        assert by_member[1] == by_member[2] == pytest.approx(merged_act)
