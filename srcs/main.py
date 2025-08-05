import os
import sys
import joblib
import numpy as np
import mne
import warnings
from scipy.signal import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from mne.io.base import BaseRaw

# Hyperparameters
TMIN = 0.0 # ok
TMAX = 4.0 # ok
L_FREQ = 12. # ok
H_FREQ = 32. # ok
BASE_SAMPLES_PER_SECONDE = 160
TARGET_SAMPLES = BASE_SAMPLES_PER_SECONDE * (TMAX - TMIN)
MOTOR_LABELS = ['C3', 'Cz', 'C4']
N_COMPONENTS = 3
MODEL_DIR = 'models'
# DATA
RUNS_LEFT_RIGHT = [3, 4, 7, 8, 11, 12]
RUNS_FIST_FEET = [5 , 6, 9, 10, 13, 14]

# ---- utils ----
def getEdfFilePath(subj, run):
    path = f"DATA/files/S{subj:03d}/S{subj:03d}R{run:02d}.edf"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return path

def getRawEDF(filepath) -> BaseRaw:
    '''
    Return a raw edf file filtered depending on hyper parameter
    '''
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    picks = [channel for channel in raw.ch_names if any(labels.lower() in channel.lower() for labels in MOTOR_LABELS)]
    if not picks:
        raise ValueError(f"No valid motor channels in {os.path.basename(filepath)}")
    raw.pick(picks, verbose=False)
    raw.filter(L_FREQ, H_FREQ, verbose=False)
    return raw

def loadModel(subj, run):
    model_file = os.path.join(MODEL_DIR, f"subj{subj:03d}_run{run:02d}.joblib")
    if not os.path.exists(model_file):
        raise Exception(f"File {model_file} not found")
    pipe = joblib.load(model_file)
    return pipe

# --- edfUtils ---

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

def train_pipeline(subj, test_run, X, y):
    pipe = Pipeline([
        ('csp', CSP(n_components=N_COMPONENTS, reg='ledoit_wolf', log=False, norm_trace=False)),
        ('scaler', StandardScaler()),
        ('lda', LDA())
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, os.path.join(MODEL_DIR, f"subj{subj:03d}_run{test_run:02d}.joblib"))
    return pipe

# ---- main ----

def init_environment():
    os.makedirs(MODEL_DIR, exist_ok=True)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    mne.set_log_level('ERROR')

def cli_train(subj, run, verbose = True):
    try:
        X, y = getAllEpochFormatedData(subj, run)
        pipe = train_pipeline(subj, run, X, y)
        if pipe is None:
            print("No data.")
            return
        scores = cross_val_score(pipe, X, y, cv=5)
        arr = np.array2string(np.round(scores, 4), separator=' ')
        if (verbose):
            print(arr)
            print(f"cross_val_score: {scores.mean():.4f}")
    except Exception as e:
        print(e)

def cli_predict(subj, run, verbose = True):
    try:
        pipe = loadModel(subj, run)
        X, y = getSingleEpochFormatedData(subj, run)
        y_pred = pipe.predict(X)
        correct = np.array(y_pred) == np.array(y)
        if (verbose):
            print("epoch nb: [prediction] [truth] equal?")
            for i, (p, t) in enumerate(zip(y_pred, y)):
                print(f"epoch {i:02d}: [{p+1}] [{t+1}] {p == t}")
            print(f"Accuracy: {correct.mean():.4f}")
        return correct.mean()
    except Exception as e:
        print(e)
        return None

def cli_all():
    accuracyGlobal = []
    for run in RUNS_LEFT_RIGHT:
        accuracyPerSubject = []
        for subj in range(1, 109):
            cli_train(subj, run, False)
            acc = cli_predict(subj, run, False)
            if acc is not None:
                accuracyPerSubject.append(acc)
                print(f"experiment {run}: subject {subj}: accuracy = {acc}")
            else:
                print(f"experiment {run}: subject {subj}: skipped (no data)")
        accuracyGlobal.append(sum(accuracyPerSubject) / len(accuracyPerSubject))
    print("Mean accuracy of the six different experiments for all 109 subjects")
    for run, acc in zip(RUNS_LEFT_RIGHT, accuracyGlobal):
        print(f"experiment {run}: accuracy = {acc}")
    print(f"Mean accuracy of {len(RUNS_LEFT_RIGHT)} experiments: {sum(accuracyGlobal) / len(accuracyGlobal)}")

if __name__ == '__main__':
    init_environment()
    if len(sys.argv) == 4 and sys.argv[3] == 'train':
        cli_train(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[3] == 'predict':
        cli_predict(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 1:
        cli_all()
    else:
        print("Usage:")
        print("  python mybci.py <subject> <run> train")
        print("  python mybci.py <subject> <run> predict")
        print("  python mybci.py")
