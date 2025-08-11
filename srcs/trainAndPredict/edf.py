import numpy as np
from .params import (
    BASE_SAMPLES_PER_SECONDE,
    RUNS_FIST_FEET,
    RUNS_LEFT_RIGHT,
    TMAX,
    TMIN,
    H_FREQ,
    L_FREQ,
)
from .utils import getEdfFilePath, getRawEDF
from mne.io.base import BaseRaw
from mne import Epochs, events_from_annotations
from scipy.signal import firwin2, filtfilt


def getEpoch(raw: BaseRaw, tmin, tmax, label):
    """Build an Epochs object for the label events."""
    events, mapping = events_from_annotations(raw, verbose=False)
    if label not in mapping:
        raise ValueError(f"No {label} annotations found.")
    epochs = Epochs(
        raw,
        events,
        event_id={label: mapping[label]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
    )
    if len(epochs) == 0:
        raise ValueError(f"No usable {label} epochs.")
    return epochs


def getGlobalFilter(raw: BaseRaw, n_taps: int = 513):
    epochs: Epochs = getEpoch(raw, TMIN, TMAX, "T0")
    nyq = BASE_SAMPLES_PER_SECONDE / 2.0
    psd = epochs.compute_psd(method="welch", fmin=0.0, fmax=nyq)
    freqs = psd.freqs
    averageFrequencies = psd.get_data().mean(axis=(0, 1))
    rawFilter = 1.0 / np.sqrt(averageFrequencies + 1e-12)
    rawFilter /= rawFilter.max()
    rawFilter[(freqs < L_FREQ) | (freqs > H_FREQ)] = 0.0
    normalizedFrequencies = freqs / nyq
    if normalizedFrequencies[0] > 0:
        normalizedFrequencies = np.insert(normalizedFrequencies, 0, 0.0)
        rawFilter = np.insert(rawFilter, 0, 0.0)
    if normalizedFrequencies[-1] < 1:
        normalizedFrequencies = np.append(normalizedFrequencies, 1.0)
        rawFilter = np.append(rawFilter, 0.0)
    finalFilter = firwin2(n_taps, normalizedFrequencies, rawFilter)
    return finalFilter


def applyGlobalFilter(raw: BaseRaw, filter):
    data = raw.get_data()
    out = raw.copy()
    out._data = filtfilt(filter, [1.0], data, axis=1)
    return out


def getRuns(test_run):
    if test_run in RUNS_LEFT_RIGHT:
        runs_to_use = RUNS_LEFT_RIGHT
    elif test_run in RUNS_FIST_FEET:
        runs_to_use = RUNS_FIST_FEET
    else:
        raise ValueError(f"Run {test_run} not in any known category")
    return runs_to_use


def extractEdfEpochs(filepath):
    raw = getRawEDF(filepath)
    h = getGlobalFilter(raw)
    raw_f = applyGlobalFilter(raw, h)

    ep_T1 = getEpoch(raw_f, TMIN, TMAX, "T1")
    ep_T2 = getEpoch(raw_f, TMIN, TMAX, "T2")

    X0 = [x for x in ep_T1.get_data()]
    X1 = [x for x in ep_T2.get_data()]

    return {0: X0, 1: X1}



def getAllEpochFormatedData(subj, test_run):
    X, y = [], []
    for run in getRuns(test_run):
        if run == test_run:
            continue
        try:
            epochs = extractEdfEpochs(getEdfFilePath(subj, run))
            X.extend(epochs[0])
            y.extend([0] * len(epochs[0]))
            X.extend(epochs[1])
            y.extend([1] * len(epochs[1]))
        except Exception as e:
            print(e)
    if not X:
        raise Exception(
            f"No data to return in getAllEpochFormatedData for subj: {subj} test_run: {test_run}"
        )
    return (X, y)


def getSingleEpochFormatedData(subj, run):
    ep = extractEdfEpochs(getEdfFilePath(subj, run))
    X = ep[0] + ep[1]
    y = [0] * len(ep[0]) + [1] * len(ep[1])
    if not X:
        raise Exception(f"No data found for subject {subj} on run {run}")
    return (X, y)
