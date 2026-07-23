"""Phase B.5 checkpoint tool: visualize ground truth for one (swaps_dir,
mz_rank) pair -- read from an already-prepared dataset if present, otherwise
compute it on the fly, so you don't need to run the full bulk build to spot-
check any single peptide (including edge cases the watershed segmentation
already got wrong).

Usage (script):
    python -m peak_detection_2d.dataset.inspect_ground_truth \
        --swaps_dir=/path/to/swaps/result --mz_rank=12345 \
        [--dataset_dir=/path/to/prepared/dataset] [--save_path=out.png]

Usage (notebook):
    from peak_detection_2d.dataset.inspect_ground_truth import visualize_ground_truth
    visualize_ground_truth(swaps_dir, mz_rank)
"""

import argparse
import logging
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .prepare_dataset import load_experiment_context, prepare_ground_truth_batch

Logger = logging.getLogger(__name__)


def _load_from_dataset(dataset_dir: str, source_experiment: str, mz_rank: int):
    """Look up (source_experiment, mz_rank) in a prepared dataset's per-split
    metadata parquet; return (image, hint, mask, split) if found, else None.
    """
    for split in ("train", "val", "test"):
        meta_path = os.path.join(dataset_dir, f"{split}_metadata.parquet")
        if not os.path.exists(meta_path):
            continue
        meta = pd.read_parquet(meta_path)
        hit = meta[
            (meta["mz_rank"] == mz_rank)
            & (meta["source_experiment"] == source_experiment)
        ]
        if hit.empty:
            continue
        row = int(hit.iloc[0]["row"])
        h5_path = os.path.join(dataset_dir, f"{split}.h5")
        with h5py.File(h5_path, "r") as f:
            image = f["images"][row]
            hint = f["hints"][row]
            mask = f["masks"][row]
        return image, hint, mask, split
    return None


def visualize_ground_truth(
    swaps_dir: str,
    mz_rank: int,
    dataset_dir: str | None = None,
    save_path: str | None = None,
):
    """Render the consensus image, hint-channel overlay, and ground-truth
    mask overlay for one peptide.

    Loads from `dataset_dir`'s prepared HDF5+parquet dataset if this
    (swaps_dir, mz_rank) pair is already in it; otherwise computes it on the
    fly via prepare_ground_truth_batch with a batch of one -- no need to run
    the full bulk build first, or to regenerate it for one-off spot checks.
    """
    source_experiment = os.path.basename(swaps_dir.rstrip("/"))
    found = None
    if dataset_dir is not None:
        found = _load_from_dataset(dataset_dir, source_experiment, mz_rank)

    if found is not None:
        image, hint, mask, split = found
        Logger.info(
            "Loaded mz_rank %s from prepared dataset %s (split=%s).",
            mz_rank,
            dataset_dir,
            split,
        )
    else:
        Logger.info(
            "mz_rank %s not found in a prepared dataset; computing on the fly.",
            mz_rank,
        )
        ctx = load_experiment_context(swaps_dir)
        results = prepare_ground_truth_batch(
            ctx["dict_ref"],
            ctx["raw_file_list"],
            swaps_dir,
            [mz_rank],
            ctx["boundary_table"],
            ctx["processing_kwargs"],
        )
        if mz_rank not in results:
            raise ValueError(
                f"mz_rank {mz_rank} has no usable ground truth in {swaps_dir} "
                "(no reference run, no known anchor in any run, or no "
                "projectable combined_ion boundary)."
            )
        dp = results[mz_rank]
        image, hint, mask = dp.image, dp.hint_channel, dp.mask

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    log_image = np.log1p(image)
    for ax in axes:
        ax.set_xlabel("IM axis")
        ax.set_ylabel("RT axis")

    axes[0].imshow(log_image, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"mz_rank {mz_rank}\nconsensus image (log)")

    axes[1].imshow(log_image, aspect="auto", origin="lower", cmap="viridis")
    hint_rc = np.argwhere(hint > 0)
    if hint_rc.size:
        axes[1].scatter(
            hint_rc[:, 1],
            hint_rc[:, 0],
            marker="*",
            color="red",
            s=150,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
    axes[1].set_title(f"hint channel ({len(hint_rc)} known anchor(s))")

    axes[2].imshow(log_image, aspect="auto", origin="lower", cmap="viridis")
    masked = np.ma.masked_where(mask == 0, mask)
    axes[2].imshow(
        masked, aspect="auto", origin="lower", cmap="Reds", alpha=0.45, vmin=0, vmax=1
    )
    axes[2].set_title(f"ground-truth mask ({int(mask.sum())} px)")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        Logger.info("Saved to %s", save_path)
    return fig


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swaps_dir", required=True)
    parser.add_argument("--mz_rank", required=True, type=int)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    visualize_ground_truth(args.swaps_dir, args.mz_rank, args.dataset_dir, args.save_path)
    if args.save_path is None:
        plt.show()


if __name__ == "__main__":
    main()
