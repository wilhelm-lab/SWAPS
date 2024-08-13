import numpy as np
import pandas as pd
from scipy import sparse, spatial
from pyteomics import mzml
from sklearn.preprocessing import normalize
import IsoSpecPy as iso
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import os
from typing import Union, Literal
from janitor import conditional_join

_AlignMethods = Literal["2stepNN", "peakRange"]
def LoadMZML(msconvert_file):
	""" read data from mzml format """
	ind,mslev,bpmz,bpint,starttime,mzarray,intarray = [],[],[],[],[],[],[]
	with mzml.read(msconvert_file) as reader:
		for each_dict in reader:
			if each_dict['ms level'] == 1:
				ind.append(each_dict['index'])
				bpmz.append(each_dict['base peak m/z'])
				bpint.append(each_dict['base peak intensity'])			
				mzarray.append(each_dict['m/z array'])
				intarray.append(each_dict['intensity array'])			
				v_dict = each_dict['scanList']
				v_dict = v_dict['scan'][0]				
				starttime.append(v_dict['scan start time'])

	mslev = [1]*len(ind)
	mzarray = [x.tolist() for x in mzarray]
	intarray = [x.tolist() for x in intarray]
	col_set = ['ind','mslev','bpmz','bpint','starttime','mzarray','intarray']
	df_ms1 = pd.DataFrame(list(zip(ind,mslev,bpmz,bpint,starttime,mzarray,intarray)), columns=col_set)
	return df_ms1


def CalcModpeptIsopattern(modpept:str, charge:int, ab_thres:float = 0.005, mod_CAM:bool = True):
    """ takes a peptide sequence with modification and charge, 
    calculate and return the two LISTs of isotope pattern with all isotopes m/z value
    with abundance larger than ab_thres, both sorted by isotope mass

    :modpept: str
    :charge: charge state of the percursor, int
    :mzrange: limitation of detection of mz range
    :mm: bin size of mz value, int
    :ab_thres: the threshold for filtering isotopes, float  

    return: two list 
    """
	
    # account for extra atoms from modification and water
    ## count extra atoms
    n_H = 2+charge # 2 from water and others from charge (proton)
    n_Mox = modpept.count('M(ox)')
    modpept = modpept.replace('(ox)', '')
    n_acetylN = modpept.count('(ac)')
    modpept = modpept.replace('(ac)', '')
    if mod_CAM:
        n_C = modpept.count('C')
    else:
        n_C = 0
    ## addition of extra atoms	
    atom_composition = iso.ParseFASTA(modpept)
    atom_composition['H'] += 3*n_C+n_H+2*n_acetylN 
    atom_composition['C'] += 2*n_C+2*n_acetylN
    atom_composition['N'] += 1*n_C
    atom_composition['O'] += 1*n_C+1+n_acetylN+1*n_Mox 

    # Isotope calculation
    formula = ''.join(["%s%s" % (key, value) for key, value in atom_composition.items()])
    iso_distr = iso.IsoThreshold(formula=formula, threshold=ab_thres, absolute=True)
    iso_distr.sort_by_mass()
    mz_sortByMass = iso_distr.np_masses()/charge
    probs_sortByMass = iso_distr.np_probs()

    return mz_sortByMass, probs_sortByMass

def AlignMZ(anchor:pd.DataFrame, precursorRow, col_to_align = ['mzarray_obs', 'mzarray_calc'], tol = 1e-4, primaryAbundanceThres:float = 0.05, AbundanceMissingThres: float = 0.4,
            method:_AlignMethods = '2stepNN', rel_height: float = 0.75, peak_results: Union[pd.DataFrame, None] = None, 
            verbose = False):
    """ align the mz column of two data frame with an error tolerance """
    sample = pd.DataFrame({'mzarray_calc':precursorRow['IsoMZ'], 'abundance':precursorRow['IsoAbundance']})
    alignment = None
    mzDelta_mean = np.nan
    mzDelta_std = np.nan
    match method:
        case '2stepNN':
            primaryIsotope = sample.loc[sample['abundance'] >= primaryAbundanceThres]
            primaryAlignment = pd.merge_asof(left=anchor.sort_values(col_to_align[0]), right=primaryIsotope.sort_values(col_to_align[1]), left_on=col_to_align[0], right_on=col_to_align[1], tolerance=tol, direction='nearest').dropna(axis=0) # type: ignore
            if primaryAlignment.shape[0] > 0: 
                primaryAlignment['alignmentRun'] = 'primary'   
                anchor = anchor[~anchor['mzarray_obs'].isin(primaryAlignment['mzarray_obs'])]
                secondaryIsotope = sample.loc[sample['abundance'] <  primaryAbundanceThres]
                secondaryAlignment = pd.merge_asof(left=anchor.sort_values(col_to_align[0]), right=secondaryIsotope.sort_values(col_to_align[1]), left_on=col_to_align[0], right_on=col_to_align[1], tolerance=tol, direction='nearest').dropna(axis=0) # type: ignore
                secondaryAlignment['alignmentRun'] = 'secondary'
                #alignment = pd.merge_asof(left=anchor.sort_values(col_to_align[0]), right=sample.sort_values(col_to_align[1]), left_on=col_to_align[0], right_on=col_to_align[1], tolerance=tol, direction='nearest').dropna(axis=0)
                alignment = pd.concat([primaryAlignment, secondaryAlignment], axis=0)
                alignment['mzDelta'] = alignment['mzarray_obs'] - alignment['mzarray_calc']
                mzDelta_mean = alignment['mzDelta'].mean()
                mzDelta_std = alignment['mzDelta'].std()

        case 'peakRange':
            if peak_results is None:
                IsoMZ = precursorRow['IsoMZ']
                anchorInRange = anchor[(anchor['mzarray_obs']>(np.min(IsoMZ)-1)) & (anchor['mzarray_obs']<(np.max(IsoMZ)+1))]
                peak_results = ExtractPeak(MZ = np.array(anchorInRange['mzarray_obs']), Intensity=np.array(anchorInRange['intensity']), rel_height=rel_height)
            alignment = conditional_join(sample, peak_results,  # type: ignore
                                         ('mzarray_calc', 'start_mz', '>='),
                                         ('mzarray_calc', 'end_mz', '<=')
                                         )
    if alignment is not None:
        IsotopeNotObs = sample[~sample['mzarray_calc'].isin(alignment['mzarray_calc'])]
        AbundanceNotObs = IsotopeNotObs['abundance'].sum()
        n_matchedIso = alignment.shape[0]

    else:
        # TODO: solve issue with multiple overlapping peak match for one isotope pattern
        IsotopeNotObs = sample
        AbundanceNotObs = 1
        n_matchedIso = 0
    IsKept = (AbundanceNotObs <= AbundanceMissingThres)
    if verbose:
        return n_matchedIso, AbundanceNotObs, IsKept, mzDelta_mean, mzDelta_std, alignment, IsotopeNotObs
    else:
        return n_matchedIso, AbundanceNotObs, IsKept, mzDelta_mean, mzDelta_std
            
def ConstructDict(CandidatePrecursorsByRT: pd.DataFrame, OneScan: Union[pd.DataFrame, pd.Series], 
                  method:_AlignMethods = '2stepNN', AbundanceMissingThres: float = 0.4, 
                  mz_tol: float = 0.01, rel_height: float = 0.75):
    MS1Intensity = pd.DataFrame({'mzarray_obs':OneScan['mzarray'], 'intensity':OneScan['intarray']})
    peak_results = None
    if method == 'peakRange':
        peak_results = ExtractPeak(np.array(MS1Intensity['mzarray_obs']), np.array(MS1Intensity['intensity']), rel_height = rel_height)
        merge_key = 'apex_mz'
        CandidateDict = peak_results[[merge_key]]
        y_true = pd.DataFrame({'mzarray_obs': peak_results['apex_mz'], 'intensity': peak_results['peak_intensity_sum']})
    else:
        merge_key = 'mzarray_obs'
        CandidateDict = MS1Intensity[[merge_key]]
        y_true = MS1Intensity
        peak_results = None
    
    # MZ alignment with row operation
    AlignmentResult = CandidatePrecursorsByRT.copy()
    AlignmentResult.loc[:, 'n_matchedIso'],  AlignmentResult.loc[:, 'AbundanceNotObs'], AlignmentResult.loc[:, 'IsKept'], AlignmentResult.loc[:, 'mzDelta_mean'], AlignmentResult.loc[:, 'mzDelta_std'], alignment, IsotopeNotObs = \
        zip(*CandidatePrecursorsByRT.apply(lambda row:AlignMZ(MS1Intensity, row, method=method, peak_results = peak_results, tol = mz_tol, verbose=True, AbundanceMissingThres=AbundanceMissingThres), axis = 1))
    
    # merge each filtered precursor into dictionary
    filteredIdx = np.where(AlignmentResult['IsKept'])[0]
    filteredPrecursorIdx = AlignmentResult[AlignmentResult['IsKept']].index
    for idx, precursor_idx in zip(filteredIdx, filteredPrecursorIdx):
        right = alignment[idx].groupby([merge_key])['abundance'].sum()
        CandidateDict = pd.merge(CandidateDict, right, on=merge_key, how='left').rename(columns={"abundance": precursor_idx}, inplace=False)
    CandidateDict = CandidateDict.groupby([merge_key]).sum()
    return CandidateDict.fillna(0), AlignmentResult, alignment, IsotopeNotObs, y_true, peak_results

def CalcPrecursorQuant(CandidateDict:pd.DataFrame, y_true:pd.DataFrame, filteredPrecursorIdx, alpha = 0.1, random_state = None):
    if alpha>0:
        mdl = linear_model.Lasso(alpha=alpha, positive=True, random_state=random_state, fit_intercept=False) # Non negative activation
    else:
        mdl = linear_model.LinearRegression(positive=True, fit_intercept=False)
    mdl.fit(X = CandidateDict[filteredPrecursorIdx], y = y_true['intensity'])
    try:
        loss = mdl.dual_gap_ # type: ignore
    except:
        loss = None
    pred = mdl.predict(X = CandidateDict[filteredPrecursorIdx])
    non_zero = np.nonzero(pred)[0]
    trueIntensitySum = y_true['intensity'].sum()
    predIntensitySum = pred.sum()
    return mdl.coef_, loss, spatial.distance.cosine(y_true['intensity'][non_zero], pred[non_zero]), pred, predIntensitySum/trueIntensitySum, len(non_zero)/y_true.shape[0]

def PlotTrueAndPredict(x, prediction, true, log:bool = False):
    if log:
        prediction = np.log10(prediction+1)
        true = np.log10(true+1)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)
    
    axs[0].vlines(x=x, ymin = 0, ymax = true)
    #sns.lineplot(x = Maxquant_result.loc[precursor_idx, 'x'], y = Maxquant_result.loc[precursor_idx, 'IsoAbundance'], ax=axs[0])
    axs[1].vlines(x = x, ymin= 0, ymax = prediction)

def PlotIsoPatternAndScan(MS1Scans, Maxquant_result,scan_idx: Union[int, None] = None,  precursor_idx:Union[int, None] = None, mzrange: Union[None, list] = None, log_intensity:bool = False, save_dir = None):
    # preprocess data

    if scan_idx is None:
        if precursor_idx is None:
            raise ValueError('Please provide a precursor index.')
        RT = Maxquant_result.loc[precursor_idx, 'Retention time']
        scan_idx = np.abs(MS1Scans['starttime'] - RT).argmin()
        scan_time = MS1Scans.loc[scan_idx, 'starttime']
        print('Precursor ', precursor_idx, 'eluted at ', RT, ', corresponding scan index ', scan_idx, 'with scan time ', scan_time)
    OneScan = MS1Scans.iloc[scan_idx, :]
    OneScanMZ = np.array(OneScan['mzarray'])
    IsoMZ = None
    if precursor_idx is not None:
        IsoMZ = Maxquant_result.loc[precursor_idx, 'IsoMZ']
        OneScanMZinRange = OneScanMZ[(OneScanMZ>(np.min(IsoMZ)-1)) & (OneScanMZ<(np.max(IsoMZ)+1))]
        OneScanMZinRangeIdx = np.where((OneScanMZ>(np.min(IsoMZ)-1)) & (OneScanMZ<(np.max(IsoMZ)+1)))[0]
    else:
        if not isinstance(mzrange, list):
            raise TypeError('mzrange should be a list, or provide an int for precursor index.')
        OneScanMZinRange = OneScanMZ[(OneScanMZ>mzrange[0]) & (OneScanMZ<mzrange[1])]
        OneScanMZinRangeIdx = np.where((OneScanMZ>mzrange[0]) & (OneScanMZ<mzrange[1]))[0]
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0)
    if log_intensity:
        Intensity = np.log10(np.array(OneScan['intarray'])[OneScanMZinRangeIdx]+1) # +1 to avoid divide by zero error
    else:
        Intensity = np.array(OneScan['intarray'])[OneScanMZinRangeIdx]

    peak_results = ExtractPeak(OneScanMZinRange, Intensity)
    peaks_idx = peak_results['apex_mzidx']
    if precursor_idx is not None:
        axs[0].vlines(x=IsoMZ, ymin = 0, ymax = Maxquant_result.loc[precursor_idx, 'IsoAbundance'])
    axs[1].vlines(x = OneScanMZinRange, ymin= -Intensity, ymax = 0)
    axs[1].hlines(y = -peak_results['peak_height'], xmin = peak_results['start_mz'], xmax = peak_results['end_mz'], linewidth = 2,color="black")
    axs[1].vlines(x = OneScanMZinRange[peaks_idx], ymin= -Intensity[peaks_idx], ymax = 0, color = 'r')
    axs[1].plot(OneScanMZinRange[peaks_idx], -Intensity[peaks_idx], "x", color = 'r')
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(fname = os.path.join(save_dir, 'SpecAndIsoPattern_scan'+str(scan_idx)+'_precursor'+str(precursor_idx)+'.png'), dpi = 300)
        plt.close()
    else:
        plt.show()
         
    return peak_results

def ExtractPeak(MZ:np.ndarray, Intensity:np.ndarray, rel_height:float = 0.75):
    peaks, _ = find_peaks(Intensity, height=0)
    (peakWidth, peakHeight, left, right) = peak_widths(Intensity, peaks, rel_height = rel_height)
    left = np.round(left, decimals=0).astype(int)
    right = np.round(right, decimals=0).astype(int)
    left_mz = MZ[left]
    right_mz = MZ[right]
    peak_intensity = [Intensity[i:j+1].sum() for (i, j) in zip(left, right)]
    peak_result = pd.DataFrame({'apex_mzidx':peaks, 
                               'apex_mz':MZ[peaks],
                               'start_mzidx':left,
                               'start_mz':left_mz,
                               'end_mzidx': right,
                               'end_mz': right_mz,
                               'peak_width':right_mz-left_mz,
                               'peak_height':peakHeight,
                               'peak_intensity_sum': peak_intensity
                               })
    return peak_result