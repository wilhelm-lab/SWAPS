import os
import subprocess
from typing import Optional, List
import pandas as pd
import numpy as np
import mokapot
import mokapot.qvalues
import mokapot.utils
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.base import BaseEstimator, clone
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

Logger = logging.getLogger(__name__)


def prepare_percolator_input(
    df,
    feature_cols: list,
    scannr_col: str = "mz_rank",
    filename_col: str = "Run_name",
    decoy_col: str = "decoy",
    peptide_col: str = "modified_sequence",
    protein_col: str = "proteins",
):
    """
    Prepares input dataframe for percolator analysis by standardizing column names and format.
    This function takes a pandas DataFrame containing PSM data and reformats it to be compatible
    with percolator's expected input format. It handles scan numbers, PSM IDs, labels for target/decoy,
    peptide sequences, and protein information.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing PSM data
    feature_cols : list
        List of column names containing features to be used for classification
    scannr_col : str, optional
        Column name containing MS/MS scan numbers (default: "mz_rank"), this column should be unique
    filename_col : str, optional
        Column name containing raw file names (default: "Run_name")
    decoy_col : str, optional
        Column name indicating decoy status (default: "decoy")
    peptide_col : str, optional
        Column name containing peptide sequences (default: "modified_sequence")
    protein_col : str, optional
        Column name containing protein identifiers (default: "proteins")
    Returns
    -------
    pandas.DataFrame
        Reformatted DataFrame with standardized column names ready for percolator input:
        id <tab> label <tab> scannr <tab> feature1 <tab> ... <tab>
            featureN <tab> peptide <tab> proteinId1 <tab> .. <tab> proteinIdM
        Labels are interpreted as 1 -- positive set and test set, -1 -- negative set.
        scannr should be integer
        id should be unique
    """
    df_pin = df.copy()
    df_pin["label"] = 1
    df_pin.loc[df_pin[decoy_col], "label"] = -1
    df_pin["id"] = (
        df_pin["mz_rank"].astype(str)
        + "_"
        + df_pin[filename_col]
        + "_"
        + df_pin[decoy_col].astype(str)
    )
    df_pin.rename(
        {scannr_col: "scannr", protein_col: "proteins", peptide_col: "peptide"},
        axis=1,
        inplace=True,
    )
    df_pin["peptide"] = "-." + df_pin["peptide"] + ".-"
    df_pin["filename"] = df_pin[filename_col]
    df_pin = df_pin[
        ["id", "label", "scannr", "filename"] + feature_cols + ["peptide", "proteins"]
    ]
    return df_pin


def prepare_mokapot_input(
    df,
    feature_cols: list,
    scannr_col: Optional[str] = None,
    psmid_col: Optional[str] = None,
    specid_col: Optional[str] = None,
    normalize_feature_cols: List[str] = [],
    decoy_col: str = "decoy",
    peptide_col: str = "modified_sequence",
    protein_col: str = "proteins",
    filename_col: Optional[str] = None,
):
    """
    Prepares input dataframe for mokapot analysis by standardizing column names and format.
    This function takes a pandas DataFrame containing PSM data and reformats it to be compatible
    with mokapot's expected input format. It handles scan numbers, PSM IDs, labels for target/decoy,
    peptide sequences, protein information, and filename data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing PSM data
    feature_cols : list
        List of column names containing features to be used for classification
    scannr_col : str, optional
        Column name containing MS/MS scan numbers (default: "MS/MS scan number"),
        this column is not necessarily unique
    psmid_col : str, optional
        Column name containing PSM IDs (default: None), this column should be unique
    normalize_features : bool, optional
        Whether to normalize feature columns (default: True)
    decoy_col : str, optional
        Column name indicating decoy status (default: "decoy")
    peptide_col : str, optional
        Column name containing peptide sequences (default: "modified_sequence")
    protein_col : str, optional
        Column name containing protein identifiers (default: "proteins")
    filename_col : str, optional
        Column name containing raw file names (default: "Raw file")

    Returns
    -------
    pandas.DataFrame
        Reformatted DataFrame with standardized column names ready for mokapot input:
        - specid: Unique identifier for each PSM
        - scannr: Scan number
        - label: Target/decoy label (1 for target, -1 for decoy)
        - [feature columns]: Selected features for classification
        - peptide: Peptide sequence
        - proteins: Protein identifiers
        - filename: Source file name
    """

    df_pin = df.copy()
    df_pin.dropna(subset=feature_cols, inplace=True)
    if scannr_col is not None:
        df_pin["scannr"] = df_pin[scannr_col]
    else:
        df_pin["scannr"] = (
            df_pin["mz_rank"].astype(str) + "_" + df_pin["label"].astype(str)
        )
    if specid_col is not None:
        df_pin["specid"] = df_pin[specid_col]
    elif psmid_col is not None:
        df_pin["specid"] = df_pin[psmid_col]
    else:
        df_pin["specid"] = (
            df_pin["mz_rank"].astype(str) + "_" + df_pin["label"].astype(str)
        )
    df_pin.drop(columns=["label"], errors="ignore", inplace=True)
    # df_pin["spectra"] = np.arange(len(df_pin))
    df_pin["label"] = 1
    df_pin.loc[df_pin[decoy_col], "label"] = -1
    id_col = ["scannr", "specid", "label", "proteins"]
    if peptide_col is not None:
        df_pin["peptide"] = df_pin[peptide_col]
        id_col += ["peptide"]
    df_pin["proteins"] = df_pin[protein_col]

    if filename_col is not None:
        df_pin["filename"] = df_pin[filename_col]
        id_col.append("filename")
    else:
        df_pin["filename"] = "file"
        id_col.append("filename")
    df_pin = df_pin[id_col + feature_cols]
    no_normalize_cols = list(set(feature_cols) - set(normalize_feature_cols))
    if len(normalize_feature_cols) > 0:
        for col in tqdm(normalize_feature_cols, desc="Normalizing features"):
            col_z = col + "_z"
            df_pin[col_z] = (df_pin[col] - df_pin[col].mean()) / df_pin[col].std()
        normalize_feature_cols = [col + "_z" for col in normalize_feature_cols]
        df_pin = df_pin[id_col + no_normalize_cols + normalize_feature_cols]
    Logger.info("Prepared mokapot input: %s", df_pin.head(5))
    return df_pin, no_normalize_cols + normalize_feature_cols


def brew_with_mokapot(
    peptide_info_dataframe: pd.DataFrame,
    train_fdr: float = 0.1,
    test_fdr: float = 0.1,
    model: Optional[BaseEstimator] = None,
    # level: Literal["pfm", "protein"] = "pfm",
    work_dir: Optional[str] = None,
    direction: Optional[str] = None,
    **kwargs,
):
    """
    Wrapper function to run mokapot rescoring on a given peptide information DataFrame.
    This function prepares the input data, runs mokapot for rescoring, and returns the results.
    Parameters
    ----------
    peptide_info_dataframe : pandas.DataFrame
        DataFrame containing peptide information and features for mokapot input
    train_fdr : float, optional
        FDR threshold for training (default: 0.1)
    test_fdr : float, optional
        FDR threshold for testing (default: 0.1)
    work_dir : str, optional
        Directory to save temporary files (default: current working directory)
    **kwargs : additional keyword arguments
        Additional arguments to pass to the prepare_mokapot_input function
    Returns
    -------
    tuple
        A tuple containing:
        - mokapot.results.PSMResults: Results of the mokapot rescoring
        - mokapot.model.Model: The trained mokapot model
    """
    if work_dir is None:
        work_dir = os.getcwd()
    else:
        os.makedirs(work_dir, exist_ok=True)
    # Prepare mokapot input
    mokapot_input, new_feature_cols = prepare_mokapot_input(
        peptide_info_dataframe,
        **kwargs,
    )
    for col in new_feature_cols:
        if col in mokapot_input.columns:
            for label, group in mokapot_input.groupby("label"):
                group[col].hist(bins=100, alpha=0.5, label=label)
            plt.legend()
            plt.savefig(
                os.path.join(work_dir, f"feature_{col}_distr.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            Logger.info(
                "Feature column %s not found in mokapot input, skipping distribution plot.",
                col,
            )
    # mokapot_input['peptide'] = mokapot_input['proteins']
    Logger.info("Mokapot input columns: %s", mokapot_input.columns.tolist())
    # Use a temporary file to write the input .pin
    mokapot_input.to_csv(
        os.path.join(work_dir, "mokapot_input.pin"), sep="\t", index=False
    )

    # Read the .pin file and run mokapot
    psms_pin = mokapot.read_pin(os.path.join(work_dir, "mokapot_input.pin"))
    if model is None:
        mokapot_model = mokapot.model.PercolatorModel(
            train_fdr=train_fdr, direction=direction
        )
    else:
        mokapot_model = mokapot.model.Model(model, train_fdr=train_fdr)
    result, model = mokapot.brew(
        psms_pin, model=mokapot_model, test_fdr=test_fdr, folds=5
    )

    # Clean up the temporary file
    # os.remove(os.path.join(work_dir, "mokapot_input.pin"))
    result.plot_qvalues()
    plt.savefig(
        os.path.join(work_dir, "mokapot_qvalues.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    result.to_txt(work_dir, decoys=True)

    # merge with mokapot results
    psms = result.confidence_estimates["psms"]
    psms_decoy = result.decoy_confidence_estimates["psms"]
    psms = pd.concat([psms, psms_decoy], ignore_index=True)
    # psms["peak_label"] = psms["specid"].str.split("_").str[1].astype(int)

    sns.histplot(data=psms, x="mokapot score", bins=100, hue="label", multiple="dodge")
    plt.savefig(
        os.path.join(work_dir, "mokapot_score_distr.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    sns.histplot(
        data=psms,
        x="mokapot score",
        hue="label",
        bins=100,
        multiple="dodge",
        log_scale=(True, False),
    )
    plt.savefig(
        os.path.join(work_dir, "mokapot_score_distr_log.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    return result, model, psms


_PERCOLATOR_POST_PROCESSING_FLAGS = {
    "tdc": "-Y",
    "mix-max": "-y",
}


def brew_with_percolator(
    peptide_info_dataframe: pd.DataFrame | None = None,
    train_fdr: float = 0.1,
    test_fdr: float = 0.1,
    work_dir: Optional[str] = None,
    post_processing: str = "tdc",
    **kwargs,
):
    """
    Wrapper function to run Percolator rescoring on a given peptide information DataFrame.

    Parameters
    ----------
    peptide_info_dataframe : pandas.DataFrame
        DataFrame containing peptide information and features for Percolator input.
    train_fdr : float, optional
        FDR threshold for training (default: 0.1). Passed as -F to percolator.
    test_fdr : float, optional
        FDR threshold for reporting results (default: 0.1). Passed as -f to percolator.
    model : BaseEstimator, optional
        Unused — kept for API symmetry with brew_with_mokapot.
    work_dir : str, optional
        Directory to write input/output files (default: current working directory).
    post_processing : str, optional
        Percolator post-processing method for assigning q-values/PEPs: "tdc"
        (target-decoy competition, passed as -Y) or "mix-max" (passed as -y).
        Default: "tdc".
    **kwargs
        Forwarded to prepare_percolator_input (e.g. feature_cols, decoy_col, peptide_col).

    Returns
    -------
    tuple
        (psms_df, peptides_df, all_psms) where psms_df contains target PSMs,
        peptides_df contains target peptides, and all_psms is target + decoy PSMs
        concatenated with a "label" column (1 = target, -1 = decoy) — mirrors the
        mokapot return convention.
    """
    if post_processing not in _PERCOLATOR_POST_PROCESSING_FLAGS:
        raise ValueError(
            f"post_processing must be one of {list(_PERCOLATOR_POST_PROCESSING_FLAGS)}, "
            f"got {post_processing!r}"
        )

    if work_dir is None:
        work_dir = os.getcwd()
    else:
        os.makedirs(work_dir, exist_ok=True)

    input_path = os.path.join(work_dir, "percolator_input.tsv")
    psms_path = os.path.join(work_dir, "percolator_psms.tsv")
    decoy_psms_path = os.path.join(work_dir, "percolator_decoy_psms.tsv")

    if not os.path.exists(input_path):
        pin_df = prepare_percolator_input(peptide_info_dataframe, **kwargs)
        pin_df.to_csv(input_path, sep="\t", index=False)

        feature_cols = kwargs.get("feature_cols", [])
        for col in feature_cols:
            if col in pin_df.columns:
                for label, group in pin_df.groupby("label"):
                    group[col].hist(bins=100, alpha=0.5, label=label)
                plt.legend()
                plt.savefig(
                    os.path.join(work_dir, f"feature_{col}_distr.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
            else:
                Logger.info(
                    "Feature column %s not found in percolator input, skipping distribution plot.",
                    col,
                )
    else:
        Logger.info(
            "Percolator input already exists, skipping preparation: %s", input_path
        )

    cmd = [
        "percolator",
        _PERCOLATOR_POST_PROCESSING_FLAGS[post_processing],
        "--only-psms",
        "-I",
        "separate",
        "--no-terminate",
        "-F",
        str(train_fdr),
        "-f",
        str(test_fdr),
        "-m",
        psms_path,
        "-M",
        decoy_psms_path,
        input_path,
    ]
    Logger.info("Running percolator: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Percolator failed (exit {e.returncode}):\n{e.stderr}"
        ) from e
    # percolator's own stdout (feature normalization stats, per-iteration progress,
    # etc.) is long enough to swamp the caller's own log/SLURM .out file -- keep it
    # on disk instead, next to the rest of this work_dir's percolator artifacts.
    percolator_log_path = os.path.join(work_dir, "percolator_run.log")
    with open(percolator_log_path, "w") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- stderr ---\n")
            f.write(proc.stderr)
    Logger.info("Percolator finished; full stdout/stderr written to %s", percolator_log_path)

    psms_df = pd.read_csv(psms_path, sep="\t")
    decoy_psms_df = pd.read_csv(decoy_psms_path, sep="\t")

    psms_df["label"] = 1
    decoy_psms_df["label"] = -1
    all_psms = pd.concat([psms_df, decoy_psms_df], ignore_index=True)

    Logger.info(
        "Percolator results: %d target PSMs, %d decoy PSMs",
        len(psms_df),
        len(decoy_psms_df),
    )

    psms_sorted = psms_df.sort_values("q-value")
    psms_sorted = psms_sorted.assign(n_accepted=np.arange(1, len(psms_sorted) + 1))
    plt.plot(psms_sorted["q-value"], psms_sorted["n_accepted"])
    plt.xlabel("q-value")
    plt.ylabel("Number of PSMs")
    plt.savefig(
        os.path.join(work_dir, "percolator_qvalues.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    sns.histplot(data=all_psms, x="score", hue="label", bins=100, multiple="dodge")
    plt.savefig(
        os.path.join(work_dir, "percolator_score_distr.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # percolator scores can be negative — log scale on count axis only
    sns.histplot(
        data=all_psms,
        x="score",
        hue="label",
        bins=100,
        multiple="dodge",
        log_scale=(False, True),
    )
    plt.savefig(
        os.path.join(work_dir, "percolator_score_distr_log.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return psms_df, None, all_psms


def select_trusted_training_rows(
    tdc_df: pd.DataFrame,
    dict_ref: pd.DataFrame,
    decoy_target_ratio: float = 1.0,
    run_col: str = "matched_run",
    rng: Optional[int] = None,
) -> pd.DataFrame:
    """Build a trusted target+decoy training pool from tdc_df.

    "Trusted" targets are rows where (mz_rank, run_col) is labelled
    Reference/Quant_Only in dict_ref, i.e. the run-peptide pair already
    has an MS/MS identification — as opposed to Not_Match candidates
    awaiting MBR, which should not be used to teach the model what a
    correct match looks like. The decoy pool (all decoy rows in
    tdc_df) is subsampled down to
    round(n_trusted_targets * decoy_target_ratio) so training starts
    from a balanced target/decoy pool, falling back to the full decoy
    pool (with a warning) if it's smaller than requested.

    Parameters
    ----------
    tdc_df : pandas.DataFrame
        Output of combine_matches_target_decoy — must have "IsTarget",
        "mz_rank", and run_col columns.
    dict_ref : pandas.DataFrame
        Reference dictionary with per-run match-status columns.
    decoy_target_ratio : float, optional
        Desired ratio of decoys to trusted targets in the training
        pool (default 1.0, i.e. balanced).
    run_col : str, optional
        Column in tdc_df identifying the run (default "matched_run").
    rng : int, optional
        Random seed for decoy subsampling.

    Returns
    -------
    pandas.DataFrame
        Concatenation of the trusted target rows and the (sub)sampled
        decoy rows.
    """
    msms_keys = get_msms_run_keys(dict_ref, run_col=run_col)
    is_msms = pd.MultiIndex.from_frame(tdc_df[["mz_rank", run_col]]).isin(
        pd.MultiIndex.from_frame(msms_keys[["mz_rank", run_col]])
    )
    trusted_targets = tdc_df.loc[tdc_df["IsTarget"] & is_msms]
    if len(trusted_targets) == 0:
        raise ValueError(
            "No MS/MS-confirmed (Reference/Quant_Only) target rows found in "
            "tdc_df — cannot build a trusted training pool."
        )
    decoy_pool = tdc_df.loc[~tdc_df["IsTarget"]]
    n_decoy_wanted = round(len(trusted_targets) * decoy_target_ratio)
    if n_decoy_wanted >= len(decoy_pool):
        if n_decoy_wanted > len(decoy_pool):
            Logger.warning(
                "select_trusted_training_rows: requested %d decoys (ratio=%s x "
                "%d trusted targets) but only %d decoys are available — using "
                "all of them.",
                n_decoy_wanted,
                decoy_target_ratio,
                len(trusted_targets),
                len(decoy_pool),
            )
        decoy_train = decoy_pool
    else:
        decoy_train = decoy_pool.sample(n=n_decoy_wanted, random_state=rng)
    Logger.info(
        "select_trusted_training_rows: %d MS/MS-trusted targets, %d decoys "
        "(pool=%d)",
        len(trusted_targets),
        len(decoy_train),
        len(decoy_pool),
    )
    return pd.concat([trusted_targets, decoy_train], ignore_index=True)


def _score_with_estimator(estimator, X: np.ndarray) -> np.ndarray:
    """Score X with `estimator`, preferring decision_function over predict_proba.

    Mirrors mokapot.model._get_scores so both model types in
    brew_trusted_target_model are scored the same way.
    """
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X)).ravel()
    proba = np.asarray(estimator.predict_proba(X))
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()


def _linear_weights(estimator, feature_names: List[str]) -> Optional[pd.Series]:
    """Feature weights for a linear estimator, or None if not linear."""
    try:
        weights = np.asarray(estimator.coef_).flatten()
        intercept = np.asarray(estimator.intercept_).flatten()
        if len(weights) != len(feature_names) or len(intercept) != 1:
            return None
    except AttributeError:
        return None
    return pd.Series(
        list(weights) + [intercept[0]], index=list(feature_names) + ["intercept"]
    )


def _plot_score_and_qvalue_diagnostics(
    df: pd.DataFrame,
    score_col: str,
    qvalue_col: str,
    label_col: str,
    work_dir: str,
    prefix: str,
) -> None:
    """Score-distribution + q-value-acceptance-curve diagnostic plots."""
    targets = df.loc[df[label_col].astype(bool)].sort_values(qvalue_col)
    plt.plot(targets[qvalue_col], np.arange(1, len(targets) + 1))
    plt.xlabel(qvalue_col)
    plt.ylabel("Number of target PSMs")
    plt.savefig(
        os.path.join(work_dir, f"{prefix}_qvalues.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    sns.histplot(data=df, x=score_col, hue=label_col, bins=100, multiple="dodge")
    plt.savefig(
        os.path.join(work_dir, f"{prefix}_score_distr.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def brew_trusted_target_model(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = "percolator",
    estimator: Optional[BaseEstimator] = None,
    train_fdr: float = 0.01,
    work_dir: Optional[str] = None,
    decoy_col: str = "Decoy",
    peptide_col: str = "Sequence",
    protein_col: str = "Proteins",
    filename_col: str = "matched_run",
    scannr_col: str = "mz_rank",
    specid_col: str = "Sequence_with_runs",
    rng: Optional[int] = None,
) -> tuple:
    """Train on a trusted (MS/MS-confirmed target + decoy) pool, score everything.

    Unlike brew_with_mokapot/brew_with_percolator (which train and
    evaluate on the same candidate pool), this trains on `train_df` —
    typically the output of select_trusted_training_rows, i.e. only
    MS/MS-confirmed targets competing against a balanced decoy sample —
    and then scores every row of `full_df` (targets and decoys alike,
    including untrusted Not_Match candidates) with the resulting model.

    Two training regimes are supported via `model_type`, differing only
    in *how* training is done — same estimator (LinearSVC), same
    class_weight hyperparameter search (mokapot.model.PERC_GRID via
    3-fold CV, exactly as mokapot.model.PercolatorModel itself sets up),
    so any difference in results is attributable to the training
    procedure, not to a different underlying classifier:
    - "percolator": mokapot's stock semi-supervised PercolatorModel —
      iterative train_fdr-based positive-label refinement — restricted
      to the trusted training pool instead of all targets.
    - "supervised": a single fit with fixed labels — MS/MS-confirmed
      target=1, decoy=0 — no iterative label refinement.
      mokapot.model.Model always runs the iterative Percolator
      algorithm regardless of the wrapped estimator, so this bypasses
      it and fits/scales a LinearSVC directly (unless `estimator` is
      given, in which case that estimator is used as-is and the
      class_weight search is skipped).

    Every row of `full_df` is scored, but before q-values are computed the
    scored population is reduced to one row per (mz_rank, matched_run) —
    the higher-scoring of the target/decoy pair at that slot, the rest
    discarded — mirroring mokapot.confidence.LinearConfidence._perform_tdc
    (mokapot.brew's own standard path) and percolator's own -Y
    postprocessing, both of which compete PSMs within the same spectrum
    before ranking. q-values are then computed via target-decoy
    competition (mokapot.qvalues.tdc) on that reduced population for both
    regimes, since mokapot's own CV/confidence machinery assumes the same
    dataset is used for training and evaluation, which does not hold
    here.

    Parameters
    ----------
    train_df, full_df : pandas.DataFrame
        tdc_df-shaped DataFrames (see combine_matches_target_decoy) —
        the trusted training pool and the full population to score.
    feature_cols : list of str
        Feature columns to use for training/scoring.
    model_type : {"percolator", "supervised"}
    estimator : sklearn estimator, optional
        Only used when model_type == "supervised". If given, used as-is
        (no class_weight search). If None (default), a LinearSVC with
        class_weight selected via the same 3-fold CV grid search
        PercolatorModel itself uses is fit directly on the trusted pool.
    train_fdr : float, optional
        Only used when model_type == "percolator".
    work_dir : str, optional
        Directory for .pin files, diagnostic plots, and weights.txt.
    decoy_col, peptide_col, protein_col, filename_col, scannr_col, specid_col : str
        Forwarded to prepare_mokapot_input. scannr_col="mz_rank" and
        specid_col="Sequence_with_runs" keep scannr a clean mz_rank
        copy and specid genuinely unique per row (tdc_df's default
        specid, mz_rank + "_" + label, collides across decoy_rep
        variants of the same mz_rank).
    rng : int, optional
        Seed for the "supervised" class_weight CV fold shuffling
        (unused for "percolator", which seeds its own CV internally).

    Returns
    -------
    tuple
        (targets_scored_df, full_scored_df, trained_model_or_estimator)
        targets_scored_df: surviving target rows (post target-decoy
            competition) with "mz_rank", "filename", "score", "q-value"
            columns — same shape brew_with_percolator produces, for
            drop-in reuse downstream.
        full_scored_df: surviving rows (targets + decoys, post target-
            decoy competition, one per mz_rank+matched_run) with "score"/
            "q-value" added, for diagnostics.
    """
    if model_type not in ("percolator", "supervised"):
        raise ValueError(
            f"model_type must be 'percolator' or 'supervised', got {model_type!r}"
        )
    if work_dir is None:
        work_dir = os.getcwd()
    else:
        os.makedirs(work_dir, exist_ok=True)

    prepare_kwargs = dict(
        feature_cols=feature_cols,
        scannr_col=scannr_col,
        specid_col=specid_col,
        decoy_col=decoy_col,
        peptide_col=peptide_col,
        protein_col=protein_col,
        filename_col=filename_col,
    )
    train_input, feat_cols = prepare_mokapot_input(train_df, **prepare_kwargs)
    full_input, _ = prepare_mokapot_input(full_df, **prepare_kwargs)
    train_pin_path = os.path.join(work_dir, f"mokapot_trusted_{model_type}_train_input.pin")
    full_pin_path = os.path.join(work_dir, f"mokapot_trusted_{model_type}_full_input.pin")
    train_input.to_csv(train_pin_path, sep="\t", index=False)
    full_input.to_csv(full_pin_path, sep="\t", index=False)

    train_psms = mokapot.read_pin(train_pin_path)
    full_psms = mokapot.read_pin(full_pin_path)

    if model_type == "percolator":
        model = mokapot.model.PercolatorModel(train_fdr=train_fdr)
        model.fit(train_psms)
        scores_full = model.predict(full_psms)
        weights = _linear_weights(model.estimator, model.features)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_psms.features.loc[:, feat_cols].values)
        y_train = train_psms.targets.astype(int)
        if estimator is None:
            # Same estimator + hyperparameter search mokapot.model.
            # PercolatorModel itself sets up (LinearSVC, class_weight chosen
            # via 3-fold CV over PERC_GRID) -- fit directly on the true
            # fixed labels here instead of PercolatorModel's iterative
            # label-refinement loop, so "percolator" vs "supervised" differ
            # only in training procedure, not model/hyperparameters.
            _rng = np.random.default_rng(rng)
            base_svm = LinearSVC(dual=False, random_state=7)
            search = GridSearchCV(
                base_svm,
                param_grid=mokapot.model.PERC_GRID,
                refit=False,
                cv=KFold(3, shuffle=True, random_state=int(_rng.integers(1, int(1e6)))),
            )
            search.fit(X_train, y_train)
            estimator = clone(base_svm).set_params(**search.best_params_)
            Logger.info(
                "brew_trusted_target_model(supervised): selected %s via 3-fold CV",
                search.best_params_,
            )
        estimator.fit(X_train, y_train)
        X_full = scaler.transform(full_psms.features.loc[:, feat_cols].values)
        scores_full = _score_with_estimator(estimator, X_full)
        weights = _linear_weights(estimator, feat_cols)
        model = estimator

    full_scored_df = full_psms.data.copy()
    full_scored_df["score"] = scores_full

    # Target-decoy competition: keep only the best-scoring PSM per (scannr,
    # filename) -- i.e. per (mz_rank, matched_run) -- before computing
    # q-values. Mirrors mokapot.confidence.LinearConfidence._perform_tdc
    # (mokapot.brew's own standard path, via mokapot.utils.groupby_max) and
    # percolator's own -Y postprocessing (verified empirically: percolator_
    # psms.tsv + percolator_decoy_psms.tsv row count equals the number of
    # unique (scannr, filename) pairs in its input, not the raw row count).
    # Skipping this step would let a target whose own paired decoy outscored
    # it keep an independent q-value here, when upstream would have dropped
    # that target from the output entirely.
    tdc_idx = mokapot.utils.groupby_max(
        full_scored_df, ["scannr", "filename"], "score", np.random.default_rng(rng)
    )
    full_scored_df = full_scored_df.loc[tdc_idx].reset_index(drop=True)
    Logger.info(
        "brew_trusted_target_model(%s): target-decoy competition kept %d of "
        "%d scored rows (best per mz_rank+matched_run).",
        model_type,
        len(full_scored_df),
        len(scores_full),
    )

    q_values_full = mokapot.qvalues.tdc(
        full_scored_df["score"].values, full_scored_df["label"].values, desc=True
    )
    full_scored_df["q-value"] = q_values_full

    if weights is not None:
        weights.to_csv(
            os.path.join(work_dir, f"mokapot_trusted_{model_type}_weights.txt"),
            sep="\t",
            header=False,
        )
    else:
        Logger.info(
            "brew_trusted_target_model(%s): estimator has no linear coef_/"
            "intercept_, skipping weights.txt.",
            model_type,
        )

    _plot_score_and_qvalue_diagnostics(
        full_scored_df,
        score_col="score",
        qvalue_col="q-value",
        label_col="label",
        work_dir=work_dir,
        prefix=f"mokapot_trusted_{model_type}",
    )

    targets_scored_df = full_scored_df.loc[full_scored_df["label"]].rename(
        columns={"scannr": "mz_rank"}
    )

    # percolator_psms.tsv-compatible scored-PSM table (PSMId/score/q-value/
    # peptide/proteinIds/filename), written next to the weights/plots -- lets
    # downstream FDR-sweep analysis code that already knows how to re-filter
    # percolator_psms.tsv at arbitrary q-value cutoffs (e.g.
    # FDR_benchmark_with_HeLa_HYE.build_filtered_combined_ions) work against
    # this trusted-target model's output too, without duplicating that logic.
    psms_tsv = targets_scored_df.copy()
    psms_tsv["PSMId"] = (
        psms_tsv["mz_rank"].astype(str)
        + "_"
        + psms_tsv["filename"].astype(str)
        + "_"
        + psms_tsv["label"].astype(str)
    )
    psms_tsv = psms_tsv.rename(columns={"proteins": "proteinIds"})[
        ["PSMId", "score", "q-value", "peptide", "proteinIds", "filename"]
    ]
    psms_tsv.to_csv(
        os.path.join(work_dir, f"mokapot_trusted_{model_type}_psms.tsv"),
        sep="\t",
        index=False,
    )

    Logger.info(
        "brew_trusted_target_model(%s): scored %d full rows (%d targets, %d "
        "decoys); %d targets pass q-value < 0.01.",
        model_type,
        len(full_scored_df),
        len(targets_scored_df),
        (~full_scored_df["label"]).sum(),
        (targets_scored_df["q-value"] < 0.01).sum(),
    )
    return targets_scored_df, full_scored_df, model


def _match_status_long(dict_ref: pd.DataFrame) -> pd.DataFrame:
    """Melt dict_ref's per-run match-status columns to long format.

    Returns a (mz_rank, Run_name, match_type) DataFrame, one row per
    (peptide, run) pair, where match_type is one of "Not_Match",
    "Reference", "Quant_Only" (or "Match"/"Other", untouched). Per-run
    columns are detected as object-dtype columns containing at least one
    of "Not_Match"/"Reference"/"Quant_Only".
    """
    file_cols = [
        col
        for col in dict_ref.columns
        if dict_ref[col].dtype == object
        and dict_ref[col].isin(["Not_Match", "Reference", "Quant_Only"]).any()
    ]
    return dict_ref[["mz_rank"] + file_cols].melt(
        id_vars="mz_rank", var_name="Run_name", value_name="match_type"
    )


def get_msms_run_keys(dict_ref: pd.DataFrame, run_col: str = "Run_name") -> pd.DataFrame:
    """(mz_rank, run_col) pairs labelled Reference/Quant_Only in dict_ref.

    These are the run-peptide pairs with an MS/MS identification (the
    run was either the peptide's global identification anchor or a
    Quant_Only run), as opposed to Not_Match candidates awaiting MBR.
    """
    long = _match_status_long(dict_ref)
    msms_keys = long.loc[
        long["match_type"].isin(["Reference", "Quant_Only"]), ["mz_rank", "Run_name"]
    ].drop_duplicates()
    if run_col != "Run_name":
        msms_keys = msms_keys.rename(columns={"Run_name": run_col})
    return msms_keys


def split_pp_by_match_status(
    dict_ref: pd.DataFrame,
    pp_match_target: pd.DataFrame,
    pp_match_decoy: pd.DataFrame,
) -> tuple:
    """Split pp_match_target/decoy into Not_Match vs Reference/Quant_Only subsets.

    Returns (pp_not_match, pp_msms, pp_decoy_not_match).  pp_not_match and
    pp_decoy_not_match contain only run-peptide pairs labelled Not_Match in
    dict_ref (candidates for MBR rescoring); pp_msms contains pairs labelled
    Reference or Quant_Only (passed through directly as MS/MS identifications).
    """
    long = _match_status_long(dict_ref)
    not_match_keys = long.loc[
        long["match_type"] == "Not_Match", ["mz_rank", "Run_name"]
    ].drop_duplicates()
    msms_keys = get_msms_run_keys(dict_ref)

    pp_not_match = pp_match_target.merge(
        not_match_keys, on=["mz_rank", "Run_name"], how="inner"
    )
    pp_msms = pp_match_target.merge(msms_keys, on=["mz_rank", "Run_name"], how="inner")
    pp_decoy_not_match = pp_match_decoy.merge(
        not_match_keys, on=["mz_rank", "Run_name"], how="inner"
    )

    Logger.info(
        "split_pp_by_match_status: %d Not_Match target, %d MS/MS target, %d Not_Match decoy",
        len(pp_not_match),
        len(pp_msms),
        len(pp_decoy_not_match),
    )
    return pp_not_match, pp_msms, pp_decoy_not_match


def combine_matches_target_decoy(matches_target, matches_decoy, dict_ref):

    ## Target Decoy Competition
    matches_target["IsTarget"] = True
    matches_decoy["IsTarget"] = False
    matches_target["decoy_mz_rank"] = -1
    tdc_df = pd.concat([matches_target, matches_decoy], ignore_index=True)
    tdc_df = pd.merge(tdc_df, dict_ref[["mz_rank", "Sequence", "Proteins"]])
    tdc_df.loc[~tdc_df["IsTarget"], "Sequence"] = (
        "DECOY_" + tdc_df.loc[~tdc_df["IsTarget"], "Sequence"]
    )
    tdc_df.loc[~tdc_df["IsTarget"], "Proteins"] = (
        "DECOY_" + tdc_df.loc[~tdc_df["IsTarget"], "Proteins"]
    )
    tdc_df["Decoy"] = ~tdc_df["IsTarget"]
    tdc_df["label"] = tdc_df["IsTarget"]
    if "feature_instance_id" not in tdc_df.columns:
        tdc_df["feature_instance_id"] = tdc_df["mz_rank"].astype(str)
    tdc_df["feature_instance_id_with_label"] = (
        tdc_df["feature_instance_id"].astype(str) + "_" + tdc_df["label"].astype(str)
    )
    if "decoy_strategy" in tdc_df.columns:
        decoy_rep = (
            tdc_df["decoy_rep"].fillna(0).astype(int).astype(str)
            if "decoy_rep" in tdc_df.columns
            else "0"
        )
        tdc_df["sequence_variant"] = np.where(
            tdc_df["Decoy"],
            tdc_df["decoy_strategy"].fillna("decoy").astype(str) + "_" + decoy_rep,
            "target_0",
        )
    else:
        tdc_df["sequence_variant"] = "target_0"
    tdc_df["Sequence_with_runs"] = (
        tdc_df["Sequence"]
        + "_"
        + tdc_df["matched_run"]
        + "_"
        + tdc_df["feature_instance_id"].astype(str)
        + "_"
        + tdc_df["sequence_variant"].astype(str)
    )
    tdc_df["feature_run"] = (
        tdc_df["feature_instance_id"].astype(str) + "_" + tdc_df["matched_run"]
    )
    return tdc_df


def normalize_shift_by_runs(
    matches_target, matches_decoy, cols_to_scale=["rt_shift", "im_shift"]
):
    """
    Normalize rt_shift and im_shift per run-pair using target statistics,
    apply normalization to both target and decoy,
    and return them separately.
    """

    # Compute mean/std per run-pair from target only
    stats = matches_target.groupby(["reference_run", "matched_run"])[cols_to_scale].agg(
        ["mean", "std"]
    )

    # Flatten MultiIndex columns
    stats.columns = [f"{col}_{agg}" for col, agg in stats.columns]

    def apply_normalization(df):
        # Merge stats onto the dataframe
        df_norm = df.merge(
            stats,
            how="left",
            left_on=["reference_run", "matched_run"],
            right_index=True,
        )

        for col in cols_to_scale:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"

            # Avoid division by zero
            df_norm[f"{col}_scaled"] = (df_norm[col] - df_norm[mean_col]) / df_norm[
                std_col
            ].replace(0, 1)
            df_norm[f"{col}_abs_scaled"] = df_norm[f"{col}_scaled"].abs()

        # Drop temporary columns
        df_norm = df_norm.drop(
            columns=[
                c for c in df_norm.columns if c.endswith("_mean") or c.endswith("_std")
            ]
        )
        return df_norm

    # Apply normalization separately
    matches_target_norm = apply_normalization(matches_target)
    matches_decoy_norm = apply_normalization(matches_decoy)

    return matches_target_norm, matches_decoy_norm
