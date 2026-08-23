"""Test/exploration CLI: MS/MS-trusted-target mokapot rescoring.

Trains a rescoring model on only the MS/MS-confirmed (Reference/Quant_Only)
target run-peptide pairs -- a trusted positive pool, since those already
passed the search engine's own FDR control -- competing against a balanced
decoy sample, then scores the full candidate population (including
untrusted Not_Match MBR candidates) with the resulting model. This is an
alternative to the production percolator path in sbs_runner_ims.py, which
trains on all target candidates (Not_Match + MS/MS-status) together.

Resumes from an existing quant_dir the same way `sbs_runner_ims --from-fdr`
does -- it does not recompute feature-feature matches.

NOT YET WIRED INTO sbs_runner_ims.py / the YACS config: there is no
cfg.FDR knob that selects this path, and sbs_runner_ims.py never imports or
calls into it. It's a standalone CLI, invoked directly (see main() below),
not reachable via a runner config alone.

Training/scoring is invariant to FDR.ONLY_SCORE_MATCH (that flag only gates
which rows are exempted from the q-value filter downstream), so each
model_type is fit once and reused for every requested only_score_match
value -- mirroring how the percolator path shares one
percolator_postprocessing_.../ dir and only nests the bool-dependent output
under only_score_match_<bool>/. Output layout:

    quant_dir/mokapot_trusted_<model_type>_trainfdr<X>/
        mokapot_trusted_<model_type>_{train,full}_input.pin
        mokapot_trusted_<model_type>_weights.txt
        mokapot_trusted_<model_type>_psms.tsv        (percolator_psms.tsv-compatible)
        mokapot_trusted_<model_type>_{qvalues,score_distr}.png
        only_score_match_<bool>/
            pp_match_target_filtered.parquet
            swaps_combined_ions.parquet
            swaps.aq_reformat.tsv* (DirectLFQ)
            result_analysis/*.png
"""

import argparse
import logging
import os

# Force the non-interactive Agg backend before anything transitively imports
# matplotlib.pyplot (postprocessing.rescore, utils.plot, ...). This is a
# batch CLI, often run detached/backgrounded on a shared node where DISPLAY
# may be set (e.g. stale SSH X11 forwarding) but no longer reachable --
# without this, matplotlib falls back to an interactive backend that raises
# an uncatchable fatal X11 IO error (bypasses Python's exception handling
# entirely) the moment that connection drops mid-run.
import matplotlib

matplotlib.use("Agg")

from utils.config import get_cfg_defaults, merge_cfg_from_file
from utils.singleton_swaps_optimization import swaps_optimization_cfg
from postprocessing.rescore import (
    select_trusted_training_rows,
    brew_trusted_target_model,
    split_pp_by_match_status,
)
from sbs_runner_ims import (
    _load_fdr_control_inputs,
    _filter_and_split_pp_by_msms,
    _build_fdr_feature_cols,
    _finalize_fdr_results,
    _to_parquet_safe,
)

# Clear existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

MODEL_TYPES = ("percolator", "supervised")
_BOOL_CHOICES = {"true": True, "false": False}


def _split_pp_for_only_score_match(dict_ref, pp_match_target, pp_match_decoy, only_score_match: bool):
    """pp_match_target_notmsms/msms for one only_score_match value.

    Mirrors the split _filter_and_split_pp_by_msms does internally, exposed
    standalone here so it can be recomputed per only_score_match value
    without re-running that helper's (idempotent but redundant) intensity
    filtering / matches_target narrowing each time.
    """
    if only_score_match:
        pp_match_target_notmsms, pp_match_target_msms, _ = split_pp_by_match_status(
            dict_ref, pp_match_target, pp_match_decoy
        )
    else:
        pp_match_target_notmsms, pp_match_target_msms = pp_match_target, None
    return pp_match_target_notmsms, pp_match_target_msms


def run_msms_trusted_rescoring(
    config_path: str,
    model_types: list,
    only_score_match_values: list,
    decoy_target_ratio: float = 1.0,
    train_fdr: float = None,
    test_fdr: float = None,
    seed: int = 0,
    exclude_features: list = None,
):
    """Resume at FDR control and rescore with MS/MS-trusted-target models.

    Runs each requested model_type (a subset of MODEL_TYPES) on the exact
    same balanced training split (same seed), so their learned weights and
    downstream results are directly comparable to one another. For each
    model_type, training/scoring happens once and its full downstream
    (q-value filtering, combined pivot, DirectLFQ, result-analysis plots)
    is produced for every requested only_score_match value -- see module
    docstring for the resulting directory layout.

    exclude_features, if given, drops those columns from the feature set
    before training/scoring -- e.g. for ablation runs. The run's files get
    a "_reduced_features" run_label/dir suffix (parent_dir_name itself is
    unchanged, i.e. it's still saved alongside a prior full-feature run
    under the same mokapot_trusted_<model_type>_trainfdr<X>/ dir) so
    nothing from an existing full-feature run in that same quant_dir gets
    overwritten.
    """
    cfg = get_cfg_defaults(swaps_optimization_cfg)  # type: ignore
    merge_cfg_from_file(cfg, config_path)
    logging.info("merge with cfg file %s", config_path)
    train_fdr = train_fdr if train_fdr is not None else cfg.FDR.TRAIN
    test_fdr = test_fdr if test_fdr is not None else cfg.FDR.TEST

    (
        processing_kwargs,
        dict_ref,
        quant_dir,
        matches_target,
        matches_decoy,
        pp_reference,
        pp_match_target,
        pp_match_decoy,
    ) = _load_fdr_control_inputs(cfg)

    # Only used here for its intensity-filtering side effect on matches_target/
    # decoy/pp_match_target/decoy (gated by FDR.ENABLED/FDR.INT_THRES, not by
    # ONLY_SCORE_MATCH) -- its own ONLY_SCORE_MATCH-gated split output is
    # discarded; that split is redone explicitly per only_score_match value
    # below via _split_pp_for_only_score_match.
    (
        matches_target,
        matches_decoy,
        pp_match_target,
        pp_match_decoy,
        _,
        _,
    ) = _filter_and_split_pp_by_msms(
        cfg, dict_ref, matches_target, matches_decoy, pp_match_target, pp_match_decoy
    )
    tdc_df, feature_cols = _build_fdr_feature_cols(
        dict_ref, processing_kwargs, matches_target, matches_decoy
    )

    if exclude_features:
        missing = [c for c in exclude_features if c not in feature_cols]
        if missing:
            logging.warning(
                "--exclude-features %s not found in the computed feature_cols "
                "%s -- ignoring those (nothing to exclude).",
                missing,
                feature_cols,
            )
        feature_cols = [c for c in feature_cols if c not in exclude_features]
        logging.info(
            "Excluded %s -- training/scoring on remaining features: %s",
            exclude_features,
            feature_cols,
        )
    dir_suffix = "_reduced_features" if exclude_features else ""

    train_df = select_trusted_training_rows(
        tdc_df, dict_ref, decoy_target_ratio=decoy_target_ratio, rng=seed
    )

    for model_type in model_types:
        logging.info(
            "=================MS/MS-trusted mokapot rescoring (%s)==================",
            model_type,
        )
        run_label = f"{model_type}{dir_suffix}"
        parent_dir_name = f"mokapot_trusted_{model_type}_trainfdr{train_fdr}"
        parent_work_dir = os.path.join(quant_dir, parent_dir_name)
        targets_scored, _, _ = brew_trusted_target_model(
            train_df,
            tdc_df,
            feature_cols,
            model_type=model_type,
            train_fdr=train_fdr,
            work_dir=parent_work_dir,
            decoy_col="Decoy",
            peptide_col="Sequence",
            protein_col="Proteins",
            filename_col="matched_run",
            rng=seed,
            run_label=run_label,
        )
        targets_filtered = targets_scored.loc[targets_scored["q-value"] < test_fdr]

        for only_score_match in only_score_match_values:
            pp_match_target_notmsms, pp_match_target_msms = _split_pp_for_only_score_match(
                dict_ref, pp_match_target, pp_match_decoy, only_score_match
            )
            dir_name = os.path.join(
                parent_dir_name, f"only_score_match_{only_score_match}{dir_suffix}"
            )
            pp_match_target_filtered = pp_match_target_notmsms.merge(
                targets_filtered[["filename", "mz_rank"]],
                left_on=["mz_rank", "Run_name"],
                right_on=["mz_rank", "filename"],
                how="inner",
            ).drop(columns=["filename"])
            out_dir = os.path.join(quant_dir, dir_name)
            os.makedirs(out_dir, exist_ok=True)
            _to_parquet_safe(
                pp_match_target_filtered,
                os.path.join(out_dir, "pp_match_target_filtered.parquet"),
                index=False,
            )

            _finalize_fdr_results(
                cfg,
                quant_dir,
                dir_name,
                pp_match_target_filtered,
                pp_match_target_msms,
                pp_reference,
                dict_ref,
            )


def _bool_or_both(value: str):
    value = value.lower()
    if value == "both":
        return "both"
    if value in _BOOL_CHOICES:
        return _BOOL_CHOICES[value]
    raise argparse.ArgumentTypeError(f"expected true/false/both, got {value!r}")


def main():
    """Entry point for the MS/MS-trusted-target mokapot rescoring CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Rescore SWAPS feature-feature matches with a mokapot model "
            "trained only on MS/MS-confirmed targets (+ a balanced decoy "
            "sample), instead of percolator's all-targets training pool."
        )
    )
    parser.add_argument("config_path", help="Path to the sbs_runner_ims configuration YAML file")
    parser.add_argument(
        "--from-fdr",
        action="store_true",
        required=True,
        help=(
            "Required: this script only supports resuming from an existing "
            "quant_dir's matches_target/decoy + pp_reference/pp_match_target/"
            "pp_match_decoy parquets (same resume semantics as "
            "`sbs_runner_ims --from-fdr`) -- it never recomputes matches."
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_TYPES) + ["both"],
        default="both",
        help=(
            "'percolator': mokapot's stock semi-supervised PercolatorModel, "
            "restricted to the MS/MS-trusted training pool. 'supervised': a "
            "plain classifier fit with fixed labels (no iterative label "
            "refinement). 'both' (default) runs both on the identical "
            "training split for direct comparison."
        ),
    )
    parser.add_argument(
        "--only-score-match",
        type=_bool_or_both,
        default="both",
        help=(
            "true/false/both (default both). Training/scoring is invariant "
            "to this flag -- each requested model_type is fit once and its "
            "downstream is produced for every requested value here, so "
            "'both' does not cost an extra model fit."
        ),
    )
    parser.add_argument(
        "--decoy-target-ratio",
        type=float,
        default=1.0,
        help="Decoys sampled per MS/MS-trusted target for training (default 1.0, balanced).",
    )
    parser.add_argument(
        "--train-fdr",
        type=float,
        default=None,
        help="Overrides cfg.FDR.TRAIN if given.",
    )
    parser.add_argument(
        "--test-fdr",
        type=float,
        default=None,
        help="Overrides cfg.FDR.TEST if given.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Decoy subsampling random seed.")
    parser.add_argument(
        "--exclude-features",
        default=None,
        help=(
            "Comma-separated feature column names to drop from training/"
            "scoring (e.g. an ablation run). Output files/dirs get a "
            "'_reduced_features' suffix so they don't overwrite a prior "
            "full-feature run in the same quant_dir."
        ),
    )
    args = parser.parse_args()

    model_types = list(MODEL_TYPES) if args.model == "both" else [args.model]
    only_score_match_values = (
        [True, False] if args.only_score_match == "both" else [args.only_score_match]
    )
    exclude_features = (
        [f.strip() for f in args.exclude_features.split(",") if f.strip()]
        if args.exclude_features
        else None
    )
    run_msms_trusted_rescoring(
        args.config_path,
        model_types,
        only_score_match_values,
        decoy_target_ratio=args.decoy_target_ratio,
        train_fdr=args.train_fdr,
        test_fdr=args.test_fdr,
        seed=args.seed,
        exclude_features=exclude_features,
    )


if __name__ == "__main__":
    main()
