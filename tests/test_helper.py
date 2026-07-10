"""
Unit tests for postprocessing.helper.build_pivot.

build_pivot takes a long-form pp_all DataFrame and dict_ref, and returns a
wide pivot with one row per mz_rank and columns for each run's Match Type
and Intensity.
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postprocessing.helper import (
    build_pivot,
    expand_group_ids_to_members,
    get_pept_act_from_parquet,
    load_peptide_batch_df_from_partquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pp_all(mz_ranks, runs, match_types, intensities):
    return pd.DataFrame(
        {
            "mz_rank": mz_ranks,
            "Run_name": runs,
            "Match Type": match_types,
            "intensity_sum": intensities,
        }
    )


def _make_dict_ref(mz_ranks):
    return pd.DataFrame({"mz_rank": list(mz_ranks)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPivot:
    def test_returns_dataframe(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_A"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_mz_rank(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_A"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        assert result.index.name == "mz_rank" or set(result.index) == {0, 1}

    def test_intensity_columns_present(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_B"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        intensity_cols = [c for c in result.columns if "Intensity" in c]
        assert len(intensity_cols) > 0

    def test_match_type_columns_present(self):
        pp = _make_pp_all([0, 1], ["run_A", "run_B"], ["MS/MS Ref", "MBR"], [1000.0, 500.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        match_cols = [c for c in result.columns if "Match Type" in c]
        assert len(match_cols) > 0

    def test_all_dict_ref_ranks_in_output(self):
        """All mz_ranks from dict_ref should appear as rows, even if no match."""
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [1000.0])
        dr = _make_dict_ref([0, 1, 2])  # ranks 1 and 2 have no match
        result = build_pivot(pp, dr)
        assert set(result.index) >= {0, 1, 2}

    def test_unmatched_rows_have_unmatched_match_type(self):
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [1000.0])
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        match_cols = [c for c in result.columns if "Match Type" in c]
        # mz_rank=1 has no entry in pp_all → should be "unmatched"
        assert result.loc[1, match_cols[0]] == "unmatched"

    def test_two_runs_produce_separate_columns(self):
        pp = _make_pp_all(
            [0, 0],
            ["run_A", "run_B"],
            ["MS/MS Ref", "MBR"],
            [1000.0, 800.0],
        )
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        run_a_int = [c for c in result.columns if "run_A" in c and "Intensity" in c]
        run_b_int = [c for c in result.columns if "run_B" in c and "Intensity" in c]
        assert len(run_a_int) == 1
        assert len(run_b_int) == 1

    def test_intensity_value_correct(self):
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [12345.0])
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        intensity_cols = [c for c in result.columns if "run_A" in c and "Intensity" in c]
        assert result.loc[0, intensity_cols[0]] == 12345.0

    def test_match_type_value_correct(self):
        pp = _make_pp_all([0], ["run_A"], ["MBR"], [500.0])
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        match_cols = [c for c in result.columns if "run_A" in c and "Match Type" in c]
        assert result.loc[0, match_cols[0]] == "MBR"

    def test_mbr_undistinguishable_group_relabeled(self):
        """MBR rows tagged with a real (non "-1") undistinguishable_group_id
        get relabeled "MBR_undistinguished" so they're excluded downstream."""
        pp = _make_pp_all([0, 1], ["run_A", "run_A"], ["MBR", "MBR"], [500.0, 600.0])
        pp["undistinguishable_group_id"] = ["10020915_0", -1]
        dr = _make_dict_ref([0, 1])
        result = build_pivot(pp, dr)
        match_col = [c for c in result.columns if "run_A" in c and "Match Type" in c][0]
        assert result.loc[0, match_col] == "MBR_undistinguished"
        assert result.loc[1, match_col] == "MBR"

    def test_non_mbr_rows_untouched_by_undistinguishable_tag(self):
        """Only "MBR" match type rows are eligible for relabeling, not
        MS/MS or MS/MS Ref rows even if flagged with a group id."""
        pp = _make_pp_all([0], ["run_A"], ["MS/MS Ref"], [1000.0])
        pp["undistinguishable_group_id"] = ["10020915_0"]
        dr = _make_dict_ref([0])
        result = build_pivot(pp, dr)
        match_col = [c for c in result.columns if "run_A" in c and "Match Type" in c][0]
        assert result.loc[0, match_col] == "MS/MS Ref"

    def test_empty_pp_all_raises_or_returns_empty(self):
        # build_pivot requires at least one row to infer run columns from
        # pivot_table — empty input produces no columns, which is acceptable
        pp = _make_pp_all([], [], [], [])
        dr = _make_dict_ref([0, 1, 2])
        try:
            result = build_pivot(pp, dr)
            # If it returns, the result should at least have the right index length
            assert len(result) == 3
        except (KeyError, ValueError):
            pass  # expected when pivot_table has nothing to work with


# ---------------------------------------------------------------------------
# expand_group_ids_to_members / load_peptide_batch_df_from_partquet
# (coSWA: lazy re-expansion of a group's single stored activation trace at
# match-features batch-read time, replacing the old SWA-write-time
# duplication that caused build_mz_sorted_activation to OOM on real data)
# ---------------------------------------------------------------------------


class TestExpandGroupIdsToMembers:
    def test_noop_when_group_to_members_empty(self):
        frame_idx = [0, 1]
        im_idx = [5, 6]
        pept_idx = [1, 2]
        data = [10.0, 20.0]
        out = expand_group_ids_to_members(frame_idx, im_idx, pept_idx, data, {})
        assert list(out[2]) == pept_idx

    def test_expands_group_row_to_all_members(self):
        group_to_members = {1001: np.array([1, 2, 3], dtype=np.uint32)}
        frame_idx = [0, 1]
        im_idx = [5, 6]
        pept_idx = [1001, 99]  # one group row, one solo real mz_rank
        data = [10.0, 20.0]
        f, i, p, d = expand_group_ids_to_members(
            frame_idx, im_idx, pept_idx, data, group_to_members
        )
        # solo row (99) untouched, group row (1001) duplicated to 3 members
        assert len(p) == 4
        solo_mask = p == 99
        assert solo_mask.sum() == 1
        assert set(p[p != 99].tolist()) == {1, 2, 3}
        # duplicated rows carry the group's original frame/im/data values
        for j in np.flatnonzero(p != 99):
            assert f[j] == 0 and i[j] == 5 and d[j] == 10.0

    def test_round_trip_recovers_identical_activation_per_member(self):
        """collapse -> solve -> expand should give every group member the
        exact same (duplicated) activation trace, since they share one row
        in the solved candidate matrix."""
        group_to_members = {1001: np.array([1, 2], dtype=np.uint32)}
        pept_idx = [1001]
        data = [42.0]
        _, _, expanded_pept, expanded_data = expand_group_ids_to_members(
            [0], [5], pept_idx, data, group_to_members
        )
        by_member = dict(zip(expanded_pept.tolist(), expanded_data.tolist()))
        assert by_member[1] == by_member[2] == 42.0


def _write_sorted_activation_parquet(act_dir, rows):
    os.makedirs(act_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(
        df[["frame_idx", "im_idx", "mz_rank", "activation"]].astype(
            {
                "frame_idx": "uint16",
                "im_idx": "uint16",
                "mz_rank": "uint32",
                "activation": "float32",
            }
        ),
        preserve_index=False,
    )
    pq.write_table(table, os.path.join(act_dir, "activation_sorted_by_mz.parquet"))
    return table


class TestLoadPeptideBatchGroupExpansion:
    """Proves the OOM fix: a coSWA group is stored on disk as exactly ONE
    row-set (not duplicated to every member), and
    load_peptide_batch_df_from_partquet still reconstructs, in memory and
    only for the requested batch, a per-member view identical to what the
    old write-time-duplicated format used to provide on disk."""

    def test_on_disk_group_row_is_not_duplicated_but_read_expands_correctly(
        self, tmp_path
    ):
        n_members = 30
        group_id = 100_000  # > any real mz_rank used below
        rows = [
            {"frame_idx": f, "im_idx": 5, "mz_rank": group_id, "activation": float(f)}
            for f in range(50)
        ]
        solo_rows = [
            {"frame_idx": f, "im_idx": 9, "mz_rank": 999, "activation": 1.0}
            for f in range(3)
        ]
        act_dir = tmp_path / "activation"
        table = _write_sorted_activation_parquet(str(act_dir), rows + solo_rows)

        # On disk: one row-set for the whole group, not n_members-fold duplicated.
        assert table.num_rows == len(rows) + len(solo_rows)

        members = list(range(1, n_members + 1))  # real mz_ranks 1..30, excludes 999
        batch_np = np.array(members + [999])
        group_to_members = {group_id: np.array(members, dtype=np.uint32)}

        df = load_peptide_batch_df_from_partquet(
            str(act_dir), batch_np, group_to_members=group_to_members
        )
        # In memory: expanded to one full row-set per member, plus solo's own rows.
        assert len(df) == len(rows) * n_members + len(solo_rows)
        for m in members:
            sub = df.loc[df["mz_rank"] == m].sort_values("frame_idx")
            assert list(sub["activation"]) == [float(f) for f in range(50)]
        assert (df.loc[df["mz_rank"] == 999, "activation"] == 1.0).all()

    def test_group_to_members_none_is_byte_identical_to_no_group_query(self, tmp_path):
        """MERGE_CONFOUNDERS.ENABLED=False path: omitting group_to_members
        (or passing None/{}) must behave exactly like before this fix."""
        rows = [
            {"frame_idx": f, "im_idx": 3, "mz_rank": mz, "activation": float(f + mz)}
            for mz in (1, 2, 3)
            for f in range(5)
        ]
        act_dir = tmp_path / "activation"
        _write_sorted_activation_parquet(str(act_dir), rows)

        batch_np = np.array([1, 2, 3])
        df_default = load_peptide_batch_df_from_partquet(str(act_dir), batch_np)
        df_explicit_none = load_peptide_batch_df_from_partquet(
            str(act_dir), batch_np, group_to_members=None
        )
        df_explicit_empty = load_peptide_batch_df_from_partquet(
            str(act_dir), batch_np, group_to_members={}
        )
        for df in (df_explicit_none, df_explicit_empty):
            pd.testing.assert_frame_equal(
                df_default.sort_values(["mz_rank", "frame_idx"]).reset_index(drop=True),
                df.sort_values(["mz_rank", "frame_idx"]).reset_index(drop=True),
            )
        assert len(df_default) == len(rows)


class TestGetPeptActGroupWindow:
    """coSWA: use_group_window crops to the merged group window (wider than the
    candidate's own individual window) for both RT and IM, with anchor offsets
    re-referenced to the group origin."""

    RUN = "runX"
    PEPT = 5

    def _act_df(self):
        # Fully populated frame 10..30 x im 0..10 so any crop is dense.
        frames, ims = np.meshgrid(np.arange(10, 31), np.arange(0, 11), indexing="ij")
        return pd.DataFrame(
            {
                "frame_idx": frames.ravel(),
                "im_idx": ims.ravel(),
                "mz_rank": self.PEPT,
                "activation": 1.0,
            }
        )

    def _dict_ref(self):
        r = self.RUN
        return pd.DataFrame(
            {
                "mz_rank": [self.PEPT],
                # individual window: RT [15,20], IM [3,6]
                f"MS1_frame_idx_left_ref_{r}": [15],
                f"MS1_frame_idx_right_ref_{r}": [20],
                f"mobility_values_index_left_ref_{r}": [3],
                f"mobility_values_index_right_ref_{r}": [6],
                # group window: RT [10,25], IM [1,8] (wider both dims)
                f"MS1_frame_idx_left_group_ref_{r}": [10],
                f"MS1_frame_idx_right_group_ref_{r}": [25],
                f"mobility_values_index_left_group_ref_{r}": [1],
                f"mobility_values_index_right_group_ref_{r}": [8],
                # experimental (predicted) anchor
                f"{r}_MS1_frame_idx_exp": [18],
                f"{r}_mobility_values_index_exp": [5],
            }
        )

    def test_individual_window_crop_and_offsets(self):
        img, rt_c, im_c = get_pept_act_from_parquet(
            self._act_df(), self.PEPT, self._dict_ref(), self.RUN
        )
        assert img.shape == (20 - 15 + 1, 6 - 3 + 1)
        assert rt_c == 18 - 15
        assert im_c == 5 - 3

    def test_group_window_is_wider_with_shifted_offsets(self):
        img, rt_c, im_c = get_pept_act_from_parquet(
            self._act_df(),
            self.PEPT,
            self._dict_ref(),
            self.RUN,
            use_group_window=True,
        )
        assert img.shape == (25 - 10 + 1, 8 - 1 + 1)
        assert rt_c == 18 - 10
        assert im_c == 5 - 1

    def test_group_window_falls_back_when_columns_absent(self):
        # No Group* columns -> use_group_window silently uses individual window.
        dr = self._dict_ref().drop(
            columns=[c for c in self._dict_ref().columns if "group_ref" in c]
        )
        img, rt_c, im_c = get_pept_act_from_parquet(
            self._act_df(), self.PEPT, dr, self.RUN, use_group_window=True
        )
        assert img.shape == (20 - 15 + 1, 6 - 3 + 1)
        assert rt_c == 18 - 15
