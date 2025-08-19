import os
import sys
import time
import numpy as np
import mne
import warnings
from sklearn.model_selection import cross_val_score
from .edf import getAllEpochFormatedData, getSingleEpochFormatedData
from .params import BASE_SAMPLES_PER_SECONDE, MODEL_DIR, RUNS_LEFT_RIGHT, MODEL_DIR, N_COMPONENTS
from .MyCSP import MyCSP
from .utils import loadModel
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def init_environment():
    os.makedirs(MODEL_DIR, exist_ok=True)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    mne.set_log_level('ERROR')

def cli_train(subj, run, verbose = True):
    try:
        X, y = getAllEpochFormatedData(subj, run)
        pipe = Pipeline([
            ('csp', MyCSP(n_components=N_COMPONENTS)),
            ('scaler', StandardScaler()),
            ('lda', LDA())
        ])
        pipe.fit(X, y)
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"subj{subj:03d}_run{run:02d}.joblib"))
        scores = cross_val_score(pipe, X, y, cv=5)
        arr = np.array2string(np.round(scores, 4), separator=' ')
        if (verbose):
            print(arr)
            print(f"cross_val_score: {scores.mean():.4f}")
    except Exception as e:
        print(e)

def simulate_stream(Xs):
    Xs = np.asarray(Xs) # garanty np
    Xf = []
    for frame in Xs.T: # Original shape: (n_channels, n_times), After transpose: (n_times, n_channels)
        Xf.append(frame)
        time.sleep(1 / BASE_SAMPLES_PER_SECONDE) # simulate stream
    return Xf

def cli_predict(subj, run):
    try:
        pipe = loadModel(subj, run)
        X, y = getSingleEpochFormatedData(subj, run)
        correct_list = []
        print("epoch nb: [prediction] [truth] equal?")
        for i, (Xs, ys) in enumerate(zip(X, y)):
            frames = simulate_stream(Xs)
            X_epoch = np.stack(frames, axis=1) # axis = 1 mean rotation
            y_pred = pipe.predict([X_epoch])[0]
            correct = (y_pred == ys)
            correct_list.append(bool(correct))
            print(f"epoch {i:02d}: [{y_pred}] [{ys}] {correct}")
        acc = float(np.mean(correct_list)) if correct_list else 0.0
        print(f"Accuracy: {acc:.4f}")
        return acc
    except Exception as e:
        print(e)
        return None

def cli_all_helper(subj, run):
    try:
        pipe = loadModel(subj, run)
        X, y = getSingleEpochFormatedData(subj, run)
        y_pred = pipe.predict(X)
        correct = np.array(y_pred) == np.array(y)
        return correct.mean()
    except Exception as e:
        print(e)
        return None

def cli_all(runs = RUNS_LEFT_RIGHT):
    accuracyGlobal = []
    for run in runs:
        accuracyPerSubject = []
        for subj in range(1, 109):
            cli_train(subj, run, False)
            acc = cli_all_helper(subj, run)
            if acc is not None:
                accuracyPerSubject.append(acc)
                print(f"experiment {run}: subject {subj}: accuracy = {acc}")
            else:
                print(f"experiment {run}: subject {subj}: skipped (no data)")
        accuracyGlobal.append(sum(accuracyPerSubject) / len(accuracyPerSubject))
    print("Mean accuracy of the six different experiments for all 109 subjects")
    for run, acc in zip(runs, accuracyGlobal):
        print(f"experiment {run}: accuracy = {acc}")
    print(f"Mean accuracy of {len(runs)} experiments: {sum(accuracyGlobal) / len(accuracyGlobal)}")

def usage():
    try:
        train = len(sys.argv) == 4 and sys.argv[3] == 'train'
        predict = len(sys.argv) == 4 and sys.argv[3] == 'predict'
        full = len(sys.argv) == 1
        if (full):
            return 1
        elif (train or predict):
            if int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 109 and int(sys.argv[2]) >= 1 and int(sys.argv[2]) <= 14:
                return 2 if train else 3
        else:
            raise Exception("Usage Error:")
    except Exception as e:
        print(e)
        print("  python mybci.py <subject> <run> train")
        print("  python mybci.py <subject> <run> predict")
        print("  python mybci.py")
    return 0

if __name__ == '__main__':
    code = usage()
    if code != 0:
        init_environment()
        if code == 2:
            cli_train(int(sys.argv[1]), int(sys.argv[2]))
        elif code == 3:
            cli_predict(int(sys.argv[1]), int(sys.argv[2]))
        elif code == 1:
            cli_all()