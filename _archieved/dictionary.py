# TODO: peakRange method is moved to Dict class, consider archieve
def ConstructDict(
    CandidatePrecursorsByRT: pd.DataFrame,
    OneScan: Union[pd.DataFrame, pd.Series],
    method: _AlignMethods = "2stepNN",
    AbundanceMissingThres: float = 0.4,
    mz_tol: float = 0.01,
    rel_height: float = 0.75,
):
    """
    Use Candidate precursors that are preselected using RT information
    to construct dictionary using isotope envelops

    TODO: add arg explanation
    """
    MS1Intensity = pd.DataFrame(
        {"mzarray_obs": OneScan["mzarray"], "intensity": OneScan["intarray"]}
    )
    peak_results = None
    logging.debug("Prepare data.")

    if method == "peakRange":
        peak_results = ExtractPeak(
            np.array(MS1Intensity["mzarray_obs"]),
            np.array(MS1Intensity["intensity"]),
            rel_height=rel_height,
        )
        merge_key = "apex_mz"
        CandidateDict = peak_results[[merge_key]]
        y_true = pd.DataFrame(
            {
                "mzarray_obs": peak_results["apex_mz"],
                "intensity": peak_results["peak_intensity_sum"],
            }
        )
        logging.debug("peak extraction")

    elif method == "2stepNN":
        merge_key = "mzarray_obs"
        CandidateDict = MS1Intensity[[merge_key]]
        y_true = MS1Intensity
        peak_results = None

    # MZ alignment with row operation
    AlignmentResult = CandidatePrecursorsByRT.copy()
    logging.info("number of row alignment %s", CandidatePrecursorsByRT.shape[0])
    (
        AlignmentResult.loc[:, "n_matchedIso"],
        AlignmentResult.loc[:, "AbundanceNotObs"],
        AlignmentResult.loc[:, "IsKept"],
        AlignmentResult.loc[:, "mzDelta_mean"],
        AlignmentResult.loc[:, "mzDelta_std"],
        alignment,
        IsotopeNotObs,
    ) = zip(
        *CandidatePrecursorsByRT.apply(
            lambda row: AlignMZ(
                MS1Intensity,
                row,
                method=method,
                # peak_results=peak_results,
                mz_tol=mz_tol,
                verbose=True,
                AbundanceMissingThres=AbundanceMissingThres,
            ),
            axis=1,
        )
    )
    logging.debug("align mz row by row.")

    # merge each filtered precursor into dictionary
    filteredIdx = np.where(AlignmentResult["IsKept"])[0]
    filteredPrecursorIdx = AlignmentResult[AlignmentResult["IsKept"]].index
    for idx, precursor_idx in zip(filteredIdx, filteredPrecursorIdx):
        right = alignment[idx].groupby([merge_key])["abundance"].sum()
        CandidateDict = pd.merge(
            CandidateDict, right, on=merge_key, how="outer"
        ).rename(columns={"abundance": precursor_idx}, inplace=False)
    logging.debug("merge dictionaries")
    CandidateDict = CandidateDict.groupby([merge_key]).sum()
    return (
        CandidateDict.fillna(0),
        AlignmentResult,
        alignment,
        IsotopeNotObs,
        y_true,
        peak_results,
    )


def AlignMZ(
    anchor: pd.DataFrame,
    precursorRow: pd.Series,
    col_to_align=["mzarray_obs", "mzarray_calc"],
    mz_tol=1e-4,
    primaryAbundanceThres: float = 0.05,
    AbundanceMissingThres: float = 0.4,
    method: _AlignMethods = "2stepNN",  # case peakRange is moved to Dict class
    verbose=False,
):
    sample = pd.DataFrame(
        {
            "mzarray_calc": precursorRow["IsoMZ"],
            "abundance": precursorRow["IsoAbundance"],
        }
    )
    alignment = None
    mzDelta_mean = np.nan
    mzDelta_std = np.nan
    match method:
        case "2stepNN":
            primaryIsotope = sample.loc[sample["abundance"] >= primaryAbundanceThres]
            primaryAlignment = pd.merge_asof(
                left=anchor.sort_values(col_to_align[0]),
                right=primaryIsotope.sort_values(col_to_align[1]),
                left_on=col_to_align[0],
                right_on=col_to_align[1],
                tolerance=mz_tol,
                direction="nearest",
            ).dropna(
                axis=0
            )  # type: ignore
            if primaryAlignment.shape[0] > 0:
                primaryAlignment["alignmentRun"] = "primary"
                anchor = anchor[
                    ~anchor["mzarray_obs"].isin(primaryAlignment["mzarray_obs"])
                ]
                secondaryIsotope = sample.loc[
                    sample["abundance"] < primaryAbundanceThres
                ]
                secondaryAlignment = pd.merge_asof(
                    left=anchor.sort_values(col_to_align[0]),
                    right=secondaryIsotope.sort_values(col_to_align[1]),
                    left_on=col_to_align[0],
                    right_on=col_to_align[1],
                    tolerance=mz_tol,
                    direction="nearest",
                ).dropna(
                    axis=0
                )  # type: ignore
                secondaryAlignment["alignmentRun"] = "secondary"
                alignment = pd.concat([primaryAlignment, secondaryAlignment], axis=0)
                alignment["mzDelta"] = (
                    alignment["mzarray_obs"] - alignment["mzarray_calc"]
                )
                mzDelta_mean = alignment["mzDelta"].mean()
                mzDelta_std = alignment["mzDelta"].std()

    if alignment is not None:
        IsotopeNotObs = sample[~sample["mzarray_calc"].isin(alignment["mzarray_calc"])]
        AbundanceNotObs = IsotopeNotObs["abundance"].sum()
        n_matchedIso = alignment.shape[0]

    else:
        IsotopeNotObs = sample
        AbundanceNotObs = 1
        n_matchedIso = 0
    IsKept = AbundanceNotObs <= AbundanceMissingThres
    if verbose:
        return (
            n_matchedIso,
            AbundanceNotObs,
            IsKept,
            mzDelta_mean,
            mzDelta_std,
            alignment,
            IsotopeNotObs,
        )
    else:
        return (
            n_matchedIso,
            AbundanceNotObs,
            IsKept,
            mzDelta_mean,
            mzDelta_std,
            None,
            None,
        )


# TODO: main method is moved to Dict class, consider archieve it


def _get_RT_edge(
    precursorRow: pd.Series,
    MS1Scans: pd.DataFrame,
    ScanIdx: int,
    direction: Literal[1, -1],
    ScanIdx_left: int,
    ScanIdx_right: int,
    IsInLastScan: Union[None, bool] = True,
    AbundanceMissingThres: float = 0.4,
):
    """
    Given a seeding scan index 'ScanIdx' that contains the target precursor,
    find the closest edge.

    :precursorRow:
    :MS1Scans:
    :ScanIdx:
    :direction: the direction for which search space is extended,
                1 stand for right range and -1 stand for left range
    :ScanIdx_left: the left edge of search limit
    :ScanIdx_right: the right edge of search limit
    :IsInLastScan: whether the precursor is in the previous scan (by time)
    :AbundanceMissingThres:

    """

    # Calculate IsInThisScan and IsInLastScan
    MS1Intensity = pd.DataFrame(
        {
            "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
            "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
        }
    )
    if IsInLastScan is None:
        _, _, IsInLastScan, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )
    if IsInLastScan:
        Logger.debug(
            "Is in scan %s and search for %s scan %s",
            ScanIdx,
            direction,
            ScanIdx + direction,
        )
        ScanIdx += direction
        MS1Intensity = pd.DataFrame(
            {
                "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
                "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
            }
        )
        _, _, IsInThisScan, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )
    else:
        if ScanIdx <= ScanIdx_right and ScanIdx >= ScanIdx_left:
            Logger.debug(
                "Is not in scan %s and search for %s scan %s",
                ScanIdx,
                -direction,
                ScanIdx - direction,
            )
            ScanIdx -= direction
            MS1Intensity = pd.DataFrame(
                {
                    "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
                    "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
                }
            )
            _, _, IsInThisScan, _, _, _, _ = AlignMZ(
                anchor=MS1Intensity,
                precursorRow=precursorRow,
                verbose=False,
                method="peakRange",
                AbundanceMissingThres=AbundanceMissingThres,
            )
        else:
            IsInThisScan = 3

    # Recursive behavior
    match (int(IsInLastScan) + int(IsInThisScan)):
        case 1:
            Logger.info("Found scan index with direction %s: %s", direction, ScanIdx)
            return ScanIdx
        case 3 | 4:
            Logger.info("Scan index out of predefined range, stop searching")
            return None
        case 0 | 2:  # consecutive N or Y
            return _get_RT_edge(
                precursorRow=precursorRow,
                MS1Scans=MS1Scans,
                ScanIdx_left=ScanIdx_left,
                ScanIdx_right=ScanIdx_right,
                direction=direction,
                ScanIdx=ScanIdx,
                IsInLastScan=IsInThisScan,
            )


def _search_BiDir_scans(
    precursorRow: pd.Series,
    MS1Scans: pd.DataFrame,
    ScanIdx: int,
    ScanIdx_left: int,
    ScanIdx_right: int,
    step: int,
    AbundanceMissingThres: float = 0.4,
):
    """
    Given a seeding scan (that does not contain target precursor),
    Search for left and right scan until target is found or reach search limit.

    :precursorRow:
    :MS1Scans:
    :ScanIdx:
    :ScanIdx_left: the left edge of search limit
    :ScanIdx_right: the right edge of search limit
    :step: the distance (+/-) between candidate scans and start scan
    :AbundanceMissingThres:
    """
    if (
        ScanIdx - step >= ScanIdx_left
    ):  # ensure search limit, only use left because of symmetricality
        MS1Intensity_next_left = pd.DataFrame(
            {
                "mzarray_obs": MS1Scans.iloc[ScanIdx - step, :]["mzarray"],
                "intensity": MS1Scans.iloc[ScanIdx - step, :]["intarray"],
            }
        )
        _, _, IsInNextLeft, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity_next_left,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )
        MS1Intensity_next_right = pd.DataFrame(
            {
                "mzarray_obs": MS1Scans.iloc[ScanIdx + step, :]["mzarray"],
                "intensity": MS1Scans.iloc[ScanIdx + step, :]["intarray"],
            }
        )
        _, _, IsInNextRight, _, _, _, _ = AlignMZ(
            anchor=MS1Intensity_next_right,
            precursorRow=precursorRow,
            verbose=False,
            method="peakRange",
            AbundanceMissingThres=AbundanceMissingThres,
        )

        match (IsInNextLeft, IsInNextRight):
            case (0, 0):
                Logger.debug(
                    "Precursor %s not observed in scan %s and %s, search with increased"
                    " step",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                step += 1
                return _search_BiDir_scans(
                    precursorRow,
                    MS1Scans,
                    ScanIdx,
                    ScanIdx_left=ScanIdx_left,
                    ScanIdx_right=ScanIdx_right,
                    step=step,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
            case (0, 1):
                Logger.debug(
                    "Precursor %s not observed in scan %s but in %s, search for right"
                    " edge",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                Left = ScanIdx + step
                Right = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx + step,
                    direction=1,
                    ScanIdx_left=ScanIdx + step,
                    ScanIdx_right=ScanIdx_right,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                return Left, Right
            case (1, 0):
                Logger.debug(
                    "Precursor %s observed in scan %s but not in %s, search for left"
                    " edge",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                Right = ScanIdx - step
                Left = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx - step,
                    direction=-1,
                    ScanIdx_left=ScanIdx_left,
                    ScanIdx_right=ScanIdx - step,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                return Left, Right
            case (1, 1):
                Logger.warning(
                    "Precursor %s observed in equal distance scan %s and %s,           "
                    "                      incorporate empty scans in the middle",
                    precursorRow["id"],
                    ScanIdx - step,
                    ScanIdx + step,
                )
                Left = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx - step,
                    direction=-1,
                    ScanIdx_left=ScanIdx_left,
                    ScanIdx_right=ScanIdx - step,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                Right = _get_RT_edge(
                    precursorRow=precursorRow,
                    MS1Scans=MS1Scans,
                    ScanIdx=ScanIdx + step,
                    direction=1,
                    ScanIdx_left=ScanIdx - step,
                    ScanIdx_right=ScanIdx_right,
                    IsInLastScan=True,
                    AbundanceMissingThres=AbundanceMissingThres,
                )
                return Left, Right
    else:
        Logger.info("Scan index out of predefined range, stop searching")
        return None, None


def locate_RT_range(
    precursorRow: pd.Series,
    MS1Scans: pd.DataFrame,
    ScanIdx: int,
    search_range: int = 100,
    step: int = 1,
    AbundanceMissingThres: float = 0.4,
):
    """
    Given a seeding scan index 'ScanIdx', find the start (side = 1) or end (side = -1) scan.

    ScanIdx has an impact on the final result, it only finds the nearest
    fulfilling condition.
    [IMPORTANT] Assumption is that ScanIdx (starting scan)
    needs to be closed enough to truth, else it will stop at the closet
    occurrence.
    Use two case scenario: whether the starting seed scan contains
    the target or not

    :precursorRow:
    :MS1Scans:
    :ScanIdx:
    :search_range: the number of scans to be searched till stop
    :AbundanceMissingThres:
    """
    ScanIdx_left = ScanIdx - search_range
    ScanIdx_right = ScanIdx + search_range
    Logger.debug(
        "Start scan = %s, Scan edge = (%s, %s)", ScanIdx, ScanIdx_left, ScanIdx_right
    )
    MS1Intensity = pd.DataFrame(
        {
            "mzarray_obs": MS1Scans.iloc[ScanIdx, :]["mzarray"],
            "intensity": MS1Scans.iloc[ScanIdx, :]["intarray"],
        }
    )

    _, _, IsInThisScan, _, _, _, _ = AlignMZ(
        anchor=MS1Intensity,
        precursorRow=precursorRow,
        verbose=False,
        method="peakRange",
        AbundanceMissingThres=AbundanceMissingThres,
    )
    if IsInThisScan:
        Logger.debug(
            "Precursor %s observed in scan %s, search for left and right edge",
            precursorRow["id"],
            ScanIdx,
        )
        Left = _get_RT_edge(
            precursorRow,
            MS1Scans,
            ScanIdx=ScanIdx - 1,
            direction=-1,
            ScanIdx_right=ScanIdx_right,
            ScanIdx_left=ScanIdx_left,
            IsInLastScan=True,
            AbundanceMissingThres=AbundanceMissingThres,
        )
        Right = _get_RT_edge(
            precursorRow,
            MS1Scans,
            ScanIdx=ScanIdx + 1,
            direction=1,
            ScanIdx_right=ScanIdx_right,
            ScanIdx_left=ScanIdx_left,
            IsInLastScan=True,
            AbundanceMissingThres=AbundanceMissingThres,
        )
        return Left, Right

    else:
        Logger.debug(
            "Precursor %s is not observed in seeding Scan %s, start searching scan %s"
            " and %s.",
            precursorRow["id"],
            ScanIdx,
            ScanIdx - step,
            ScanIdx + step,
        )
        return _search_BiDir_scans(
            precursorRow=precursorRow,
            MS1Scans=MS1Scans,
            ScanIdx=ScanIdx,
            ScanIdx_left=ScanIdx_left,
            ScanIdx_right=ScanIdx_right,
            step=step,
            AbundanceMissingThres=AbundanceMissingThres,
        )
