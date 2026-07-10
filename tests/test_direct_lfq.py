"""
Unit tests for postprocessing.direct_lfq.reformat_swaps_combined_for_directlfq.
"""

import os
import numpy as np
import pandas as pd

from postprocessing.direct_lfq import (
    reformat_swaps_combined_for_directlfq,
    undistinguishable_excl_output_name,
)


class TestReformatExcludesUndistinguishableMBR:
    def test_mbr_undistinguished_intensity_excluded(self, tmp_path):
        combined_ion = pd.DataFrame(
            {
                "mz_rank": [0, 1],
                "run_A Match Type": ["MBR_undistinguished", "MBR"],
                "run_A Intensity": [500.0, 600.0],
            }
        )
        dict_ref = pd.DataFrame({"mz_rank": [0, 1], "Proteins": ["P1", "P2"]})
        result = reformat_swaps_combined_for_directlfq(
            combined_ion, dict_ref, output_dir=str(tmp_path)
        )
        by_ion = result.set_index("ion")["run_run_A Intensity"]
        assert np.isnan(by_ion.loc[0])
        assert by_ion.loc[1] == 600.0


class TestReformatUndistinguishableGroupIdSplit:
    def _combined_ion(self):
        return pd.DataFrame(
            {
                "mz_rank": [0, 1, 2, 3],
                "run_A Intensity": [500.0, 600.0, 700.0, 800.0],
                "undistinguishable_group_id": ["10020915_0", "10020915_0", -1, np.nan],
            }
        )

    def _dict_ref(self):
        return pd.DataFrame(
            {"mz_rank": [0, 1, 2, 3], "Proteins": ["P1", "P2", "P3", "P4"]}
        )

    def test_writes_both_versions_when_other_tags_present(self, tmp_path):
        result = reformat_swaps_combined_for_directlfq(
            self._combined_ion(), self._dict_ref(), output_dir=str(tmp_path)
        )
        # default output_name: unfiltered "include" version keeps all 4 ions
        assert set(result["ion"]) == {0, 1, 2, 3}
        assert os.path.exists(os.path.join(tmp_path, "swaps.aq_reformat.tsv"))

        excl_path = os.path.join(tmp_path, "swaps.aq_reformat_excl_undistinguishable.tsv")
        assert os.path.exists(excl_path)
        excl_df = pd.read_csv(excl_path, sep="\t")
        # mz_rank 0 and 1 (tagged "10020915_0") dropped; -1 and NaN rows kept
        assert set(excl_df["ion"]) == {2, 3}

    def test_no_split_file_when_all_untagged(self, tmp_path):
        combined_ion = self._combined_ion()
        combined_ion["undistinguishable_group_id"] = [-1, -1, -1, np.nan]
        reformat_swaps_combined_for_directlfq(
            combined_ion, self._dict_ref(), output_dir=str(tmp_path)
        )
        excl_path = os.path.join(tmp_path, "swaps.aq_reformat_excl_undistinguishable.tsv")
        assert not os.path.exists(excl_path)

    def test_no_group_id_column_behaves_as_before(self, tmp_path):
        combined_ion = self._combined_ion().drop(columns=["undistinguishable_group_id"])
        result = reformat_swaps_combined_for_directlfq(
            combined_ion, self._dict_ref(), output_dir=str(tmp_path)
        )
        assert "undistinguishable_group_id" not in result.columns
        excl_path = os.path.join(tmp_path, "swaps.aq_reformat_excl_undistinguishable.tsv")
        assert not os.path.exists(excl_path)

    def test_excl_output_name_matches_written_file(self, tmp_path):
        """sbs_runner_ims.py derives the excluded-version path from this
        helper to decide whether to run DirectLFQ on it again -- must match
        exactly what reformat_swaps_combined_for_directlfq actually writes."""
        output_name = "swaps.aq_reformat.tsv"
        reformat_swaps_combined_for_directlfq(
            self._combined_ion(),
            self._dict_ref(),
            output_dir=str(tmp_path),
            output_name=output_name,
        )
        excl_path = os.path.join(
            tmp_path, undistinguishable_excl_output_name(output_name)
        )
        assert os.path.exists(excl_path)
