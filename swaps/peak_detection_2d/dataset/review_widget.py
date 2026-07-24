"""Interactive Jupyter review widget for gt_correction.py's watershed-label
selection + polygon-override workflow (see that module's docstring for the
full design rationale).

Requires `%matplotlib widget` (the `ipympl` package) in the notebook cell
before instantiating GroundTruthReviewSession -- the default inline/Agg
backend can't receive click events, and this works over a remote/browser
Jupyter session the same way ipywidgets itself already does (no X11/Qt
display needed). Install with `pip install ipympl` if `%matplotlib widget`
errors.

Usage (in a notebook, after `%matplotlib widget`):

    from peak_detection_2d.dataset.gt_correction import (
        ReviewStore, sample_diverse_review_batch,
    )
    from peak_detection_2d.dataset.review_widget import GroundTruthReviewSession

    swaps_dirs = [swaps_dir_1, swaps_dir_2]
    plan = sample_diverse_review_batch(swaps_dirs, n_samples=300)
    store = ReviewStore(output_dir="/path/to/gt_review", reviewer="your_name")
    swaps_dirs_by_experiment = {os.path.basename(d.rstrip("/")): d for d in swaps_dirs}
    session = GroundTruthReviewSession(plan, swaps_dirs_by_experiment, store)
    session.show()

Constructing the session does all of the plan's per-peptide consensus-image
computation up front, batched per experiment (see _preload_datapoints) --
this can take a little while for a few hundred samples, but keeps every
subsequent click/accept instant; see that method's docstring for why this
matters more than the plotting itself.

Controls: click a colored region to toggle it in/out of the kept set
(default selection = every region touching a known anchor, the star
markers). "Reset to default" restores that heuristic selection. "Polygon
mode" switches to click-to-place-vertices freehand override (close near the
first vertex) for the minority of cases where no label union matches the
true (possibly irregular) footprint. "Discard sample" drops a sample
entirely -- expected occasionally, since the crop is never padded beyond the
existing bbox (see gt_correction module docstring). "Accept & next" commits
whichever mode produced the current mask and advances; progress is saved to
`store` after every decision, so closing and re-running with the same
`plan` resumes where you left off (already-reviewed samples are also
skipped by _preload_datapoints, not just by the stepper).
"""

import logging

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.widgets import PolygonSelector
from skimage.segmentation import find_boundaries

from .gt_correction import (
    ReviewStore,
    compute_watershed_crop,
    default_kept_label_ids,
    extract_bbox_from_mask,
    mask_from_label_ids,
    mask_from_polygon_vertices,
)
from .prepare_dataset import GroundTruthDatapoint, load_experiment_context, prepare_ground_truth_batch

Logger = logging.getLogger(__name__)


class GroundTruthReviewSession:
    """Steps through a review plan (source_experiment, mz_rank) one sample
    at a time -- see module docstring for controls and setup."""

    def __init__(
        self,
        plan: pd.DataFrame,
        swaps_dirs_by_experiment: dict[str, str],
        store: ReviewStore,
        preload_batch_size: int = 500,
    ):
        self.store = store
        self.swaps_dirs_by_experiment = swaps_dirs_by_experiment
        todo_mask = ~plan.apply(
            lambda r: store.is_done(r["source_experiment"], int(r["mz_rank"])), axis=1
        )
        self.plan = plan[todo_mask].reset_index(drop=True)
        n_done = len(plan) - len(self.plan)
        if n_done:
            Logger.info("Resuming: %d/%d already reviewed, skipping.", n_done, len(plan))

        self._contexts: dict[str, dict] = {}
        self._datapoints = self._preload_datapoints(preload_batch_size)
        self._index = 0
        self._kept_ids: set[int] = set()
        self._polygon_mode = False
        self._polygon_selector = None
        self._polygon_verts: list[tuple[float, float]] | None = None
        self._current: dict | None = None
        self._base_im = None
        self._overlay_im = None
        self._contour = None
        self._hint_scatter = None

        self._progress_label = widgets.Label()
        self._status_label = widgets.Label()
        self._reset_btn = widgets.Button(description="Reset to default")
        self._polygon_btn = widgets.ToggleButton(description="Polygon mode")
        self._discard_btn = widgets.Button(description="Discard sample", button_style="danger")
        self._accept_btn = widgets.Button(description="Accept & next", button_style="success")

        self._reset_btn.on_click(self._on_reset)
        self._polygon_btn.observe(self._on_polygon_toggle, names="value")
        self._discard_btn.on_click(self._on_discard)
        self._accept_btn.on_click(self._on_accept)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _context(self, source_experiment: str) -> dict:
        if source_experiment not in self._contexts:
            self._contexts[source_experiment] = load_experiment_context(
                self.swaps_dirs_by_experiment[source_experiment]
            )
        return self._contexts[source_experiment]

    def _preload_datapoints(
        self, batch_size: int
    ) -> dict[tuple[str, int], GroundTruthDatapoint]:
        """Batch-computes every remaining plan sample's consensus
        image/hint/mask up front, grouped by experiment and chunked at
        `batch_size` -- one prepare_ground_truth_batch call per chunk
        instead of one per sample.

        This used to happen lazily, one mz_rank at a time, inside
        _load_current on every single "accept & next" -- which defeated the
        whole point of prepare_ground_truth_batch's own batched-I/O design
        (it exists specifically so the DuckDB activation scan happens once
        per batch, not once per peptide -- see prepare_dataset.py's module
        docstring). That per-click reload, not the ipympl plotting, was the
        main source of lag: it re-ran the full alignment/denoise pipeline
        for one peptide on every click instead of once per review session.
        """
        datapoints: dict[tuple[str, int], GroundTruthDatapoint] = {}
        for source_experiment, sub in self.plan.groupby("source_experiment"):
            ctx = self._context(source_experiment)
            swaps_dir = self.swaps_dirs_by_experiment[source_experiment]
            mz_ranks = sub["mz_rank"].astype(int).to_numpy()
            n_found = 0
            for start in range(0, len(mz_ranks), batch_size):
                chunk = mz_ranks[start : start + batch_size]
                batch_results = prepare_ground_truth_batch(
                    ctx["dict_ref"],
                    ctx["raw_file_list"],
                    swaps_dir,
                    chunk,
                    ctx["boundary_table"],
                    ctx["processing_kwargs"],
                )
                for mz_rank, dp in batch_results.items():
                    datapoints[(source_experiment, mz_rank)] = dp
                n_found += len(batch_results)
            Logger.info(
                "Preloaded %d/%d samples for %s", n_found, len(mz_ranks), source_experiment
            )
        return datapoints

    def show(self):
        controls = widgets.HBox(
            [self._reset_btn, self._polygon_btn, self._discard_btn, self._accept_btn]
        )
        display(
            widgets.VBox([self._progress_label, self.fig.canvas, controls, self._status_label])
        )
        self._load_current()

    def _load_current(self):
        if self._index >= len(self.plan):
            self._status_label.value = "All done -- nothing left to review in this plan."
            self._current = None
            self.ax.clear()
            self.fig.canvas.draw_idle()
            return

        plan_row = self.plan.iloc[self._index]
        source_experiment, mz_rank = plan_row["source_experiment"], int(plan_row["mz_rank"])
        dp = self._datapoints.get((source_experiment, mz_rank))
        if dp is None:
            Logger.warning(
                "mz_rank %s (%s) has no usable ground truth; skipping.",
                mz_rank,
                source_experiment,
            )
            self._index += 1
            self._load_current()
            return

        ctx = self._context(source_experiment)
        peak_consensus_kwargs = dict(ctx["processing_kwargs"].get("peak_consensus_kwargs", {}))
        bbox = extract_bbox_from_mask(dp.mask)
        crop, labels = compute_watershed_crop(dp.image, bbox, peak_consensus_kwargs)
        row0, col0, row1, col1 = bbox
        hint_crop = dp.hint_channel[row0:row1, col0:col1]

        self._current = {
            "source_experiment": source_experiment,
            "mz_rank": mz_rank,
            "dp": dp,
            "crop": crop,
            "labels": labels,
            "hint_crop": hint_crop,
            "bbox": bbox,
        }
        self._kept_ids = default_kept_label_ids(labels, hint_crop)
        self._polygon_mode = False
        self._polygon_btn.value = False
        self._polygon_verts = None
        self._progress_label.value = (
            f"{self._index + 1}/{len(self.plan)} -- {source_experiment} / mz_rank {mz_rank}"
        )
        n_regions = int(labels.max())
        self._status_label.value = f"{n_regions} watershed region(s); default = touches-hint."
        self._draw_base()
        self._update_overlay()

    def _draw_base(self):
        """Draws everything that only depends on which *sample* is loaded
        (base image, region boundaries, hint markers) -- called once per
        sample. Toggling a label only needs the cheap _update_overlay()
        below instead of a full re-render: every ipympl redraw round-trips
        a full frame over the notebook's websocket, so avoiding a full
        ax.clear()-and-redraw on every click is the other main lever for
        keeping the session responsive (see _preload_datapoints for the
        bigger one)."""
        cur = self._current
        labels = cur["labels"]
        self.ax.clear()
        self._base_im = self.ax.imshow(
            np.log1p(cur["crop"]), origin="lower", aspect="auto", cmap="viridis"
        )
        self._contour = self.ax.contour(
            find_boundaries(labels, mode="outer"), colors="cyan", linewidths=0.5
        )
        hint_rc = np.argwhere(cur["hint_crop"] > 0)
        self._hint_scatter = None
        if hint_rc.size:
            self._hint_scatter = self.ax.scatter(
                hint_rc[:, 1], hint_rc[:, 0], marker="*", color="white", s=120,
                edgecolor="black", linewidth=0.8, zorder=5,
            )
        empty = np.ma.masked_all(labels.shape, dtype=float)
        self._overlay_im = self.ax.imshow(
            empty, origin="lower", aspect="auto", cmap="Reds", alpha=0.5, vmin=0, vmax=1
        )
        self.fig.canvas.draw_idle()

    def _update_overlay(self):
        """Cheap per-toggle update: only touches the kept-region overlay's
        data and the label-vs-polygon-mode artist visibility, no
        base-image/contour/scatter recompute."""
        cur = self._current
        labels = cur["labels"]
        label_mode_visible = not self._polygon_mode
        for coll in self._contour.collections:
            coll.set_visible(label_mode_visible)
        if self._polygon_mode:
            self._overlay_im.set_visible(False)
            self.ax.set_title("click vertices, close near start")
        else:
            kept_mask = (
                np.isin(labels, list(self._kept_ids))
                if self._kept_ids
                else np.zeros_like(labels, dtype=bool)
            )
            masked = np.ma.masked_where(~kept_mask, kept_mask.astype(float))
            self._overlay_im.set_data(masked)
            self._overlay_im.set_visible(True)
            self.ax.set_title("click a region to toggle")
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if self._polygon_mode or self._current is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        r, c = int(round(event.ydata)), int(round(event.xdata))
        labels = self._current["labels"]
        if not (0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]):
            return
        lbl = int(labels[r, c])
        if lbl == 0:
            return
        self._kept_ids.symmetric_difference_update({lbl})
        self._update_overlay()

    def _on_reset(self, _btn):
        if self._current is None:
            return
        self._kept_ids = default_kept_label_ids(self._current["labels"], self._current["hint_crop"])
        self._update_overlay()

    def _on_polygon_toggle(self, change):
        self._polygon_mode = bool(change["new"])
        if self._current is None:
            return
        self._update_overlay()
        if self._polygon_mode:
            self._polygon_verts = None
            self._polygon_selector = PolygonSelector(self.ax, onselect=self._on_polygon_select)
        elif self._polygon_selector is not None:
            self._polygon_selector.disconnect_events()
            self._polygon_selector = None

    def _on_polygon_select(self, verts):
        self._polygon_verts = list(verts)
        self._status_label.value = f"Polygon: {len(verts)} vertices captured."

    def _on_discard(self, _btn):
        if self._current is None:
            return
        self.store.record_discarded(self._current["source_experiment"], self._current["mz_rank"])
        self._advance()

    def _on_accept(self, _btn):
        cur = self._current
        if cur is None:
            return
        full_shape = cur["dp"].mask.shape
        if self._polygon_mode:
            if not self._polygon_verts or len(self._polygon_verts) < 3:
                self._status_label.value = (
                    "Draw a polygon (>= 3 vertices) before accepting, or turn polygon mode off."
                )
                return
            corrected_mask = mask_from_polygon_vertices(self._polygon_verts, cur["bbox"], full_shape)
            method = "polygon"
        else:
            if not self._kept_ids:
                self._status_label.value = "No regions selected -- click at least one, or use Discard."
                return
            corrected_mask = mask_from_label_ids(cur["labels"], self._kept_ids, cur["bbox"], full_shape)
            method = "labels"

        self.store.record_reviewed(
            cur["source_experiment"],
            cur["mz_rank"],
            method,
            crop_image=cur["crop"],
            watershed_labels=cur["labels"],
            hint_crop=cur["hint_crop"],
            bbox=cur["bbox"],
            full_shape=full_shape,
            corrected_mask=corrected_mask,
        )
        self._advance()

    def _advance(self):
        if self._polygon_selector is not None:
            self._polygon_selector.disconnect_events()
            self._polygon_selector = None
        self._index += 1
        self._load_current()
