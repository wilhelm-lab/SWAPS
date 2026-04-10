"""
Unit tests for postprocessing.match_features.
"""

import numpy as np

from postprocessing.match_features import quantify_from_coords, QuantificationResult


class TestQuantifyFromCoordsBounds:
    def test_anchor_out_of_bounds_no_reference(self):
        image = np.zeros((10, 10))

        # Beyond image shape
        res = quantify_from_coords(image, template_anchor=(15, 5))
        assert not res.succeeded

        # Negative coordinate
        res2 = quantify_from_coords(image, template_anchor=(-2, 5))
        assert not res2.succeeded

    def test_anchor_out_of_bounds_with_reference(self):
        image = np.zeros((10, 10))
        ref_image = np.zeros((5, 5))

        # Beyond reference image shape
        res = quantify_from_coords(
            image, template_anchor=(6, 2), reference_image=ref_image
        )
        assert not res.succeeded

        # Negative coordinate
        res2 = quantify_from_coords(
            image, template_anchor=(2, -1), reference_image=ref_image
        )
        assert not res2.succeeded
