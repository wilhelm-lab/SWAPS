"""
Unit tests for postprocessing.direct_lfq.reformat_swaps_combined_for_directlfq.
"""

import numpy as np
import pandas as pd

from postprocessing.direct_lfq import reformat_swaps_combined_for_directlfq


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
