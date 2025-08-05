from params import RUNS_FIST_FEET, RUNS_LEFT_RIGHT, TARGET_SAMPLES, TMAX, TMIN
from utils import getEdfFilePath, getRawEDF

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
    epochs = {0: [], 1: []}
    duration = raw.times[-1]
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        if description not in ('T1', 'T2'):
            continue
        label = 0 if description == 'T1' else 1
        start, end = onset + TMIN, onset + TMAX
        if start < 0 or end > duration:
            continue
        try:
            data = raw.copy().crop(start, end, include_tmax=False).get_data()
            if data.shape[1] != TARGET_SAMPLES:
                continue
            epochs[label].append(data)
        except Exception as e:
            print(e)
    return epochs

def getAllEpochFormatedData(subj, test_run):
    X, y = [], []
    for run in getRuns(test_run):
        if run == test_run:
            continue
        try:
            epoch = extractEdfEpochs(getEdfFilePath(subj, run))
            for lbl in (0, 1):
                X.extend(epoch[lbl])
                y.extend([lbl] * len(epoch[lbl]))
        except Exception as e:
            print(e)
    if not X:
        raise Exception(f"No data to return in getAllEpochFormatedData for subj: {subj} test_run: {test_run}")
    return (X, y)

def getSingleEpochFormatedData(subj, run):
    ep = extractEdfEpochs(getEdfFilePath(subj, run))
    X = ep[0] + ep[1]
    y = [0] * len(ep[0]) + [1] * len(ep[1])
    if not X:
        raise Exception(f"No data found for subject {subj} on run {run}")
    return (X, y)
