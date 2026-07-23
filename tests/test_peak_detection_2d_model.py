import copy

import torch

from swaps.peak_detection_2d.config.singleton_peak_detection import peak_detection_cfg
from swaps.peak_detection_2d.loss.build_criterion import build_criterion
from swaps.peak_detection_2d.loss.combo_loss import per_image_weighted_iou_metric
from swaps.peak_detection_2d.model.build_model import build_model


def _cfg():
    return copy.deepcopy(peak_detection_cfg)


def test_build_model_forward_pass_shape():
    cfg = _cfg()
    model = build_model(cfg.MODEL)
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, cfg.MODEL.PARAMS.IN_CHANNELS, 112, 528)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 1, 112, 528)


def test_combo_loss_backward_pass():
    cfg = _cfg()
    model = build_model(cfg.MODEL)
    criterion = build_criterion(cfg.MODEL.SOLVER.LOSS)

    batch_size = 2
    x = torch.randn(batch_size, cfg.MODEL.PARAMS.IN_CHANNELS, 112, 528)
    target = (torch.rand(batch_size, 1, 112, 528) > 0.5).float()

    out = model(x)
    loss = criterion(out, target, x)
    loss.backward()

    assert torch.isfinite(loss)
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert grad_norms, "no gradients were populated by backward()"
    assert any(g > 0 for g in grad_norms)


def _logits_for_mask(mask: torch.Tensor) -> torch.Tensor:
    """Large-magnitude logits that sigmoid-threshold back to exactly `mask`."""
    return torch.where(mask > 0.5, torch.full_like(mask, 10.0), torch.full_like(mask, -10.0))


class TestPerImageWeightedIouMetric:
    def test_perfect_prediction_scores_one(self):
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:5, 2:5] = 1.0
        image = torch.rand(1, 3, 8, 8)  # channel 0 selected as the weight
        score = per_image_weighted_iou_metric(
            _logits_for_mask(mask), mask, image, threshold=0.5, device="cpu", channel=0
        )
        assert torch.allclose(score, torch.ones(1), atol=1e-4)

    def test_disjoint_prediction_scores_near_zero(self):
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0:2, 0:2] = 1.0
        pred_mask = torch.zeros(1, 1, 8, 8)
        pred_mask[:, :, 6:8, 6:8] = 1.0
        image = torch.ones(1, 3, 8, 8)
        score = per_image_weighted_iou_metric(
            _logits_for_mask(pred_mask), mask, image, threshold=0.5, device="cpu", channel=0
        )
        assert score.item() < 1e-3

    def test_intensity_weighting_favors_high_intensity_overlap(self):
        # Two candidate predictions cover equal-area, non-overlapping halves
        # of the true mask; only the weighting image differs, concentrating
        # intensity under pred_a's half vs pred_b's half.
        mask = torch.zeros(1, 1, 4, 8)
        mask[:, :, :, :] = 1.0  # full image is target
        pred_a = torch.zeros(1, 1, 4, 8)
        pred_a[:, :, :, :4] = 1.0  # left half
        pred_b = torch.zeros(1, 1, 4, 8)
        pred_b[:, :, :, 4:] = 1.0  # right half

        image = torch.ones(1, 3, 4, 8) * 0.01
        image[:, 0, :, :4] = 10.0  # channel 0: high intensity on the left half

        score_a = per_image_weighted_iou_metric(
            _logits_for_mask(pred_a), mask, image, threshold=0.5, device="cpu", channel=0
        )
        score_b = per_image_weighted_iou_metric(
            _logits_for_mask(pred_b), mask, image, threshold=0.5, device="cpu", channel=0
        )
        assert score_a.item() > score_b.item()

    def test_batched_and_per_image_shapes(self):
        mask = (torch.rand(3, 1, 6, 6) > 0.5).float()
        image = torch.rand(3, 3, 6, 6)
        score = per_image_weighted_iou_metric(
            _logits_for_mask(mask), mask, image, threshold=0.5, device="cpu", channel=0
        )
        assert score.shape == (3,)
        # Every value is a well-formed IoU in [0, 1] (thresholded pred exactly
        # matches target here, so all three should be ~1).
        assert torch.all(score > 0.99)
