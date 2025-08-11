import os
import joblib
import mne
from mne.io.base import BaseRaw
from .params import H_FREQ, L_FREQ, MODEL_DIR, MOTOR_LABELS

def getEdfFilePath(subj, run):
    path = f"DATA/files/S{subj:03d}/S{subj:03d}R{run:02d}.edf"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return path

def getRawEDF(filepath) -> BaseRaw:
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    return raw

def loadModel(subj, run):
    model_file = os.path.join(MODEL_DIR, f"subj{subj:03d}_run{run:02d}.joblib")
    if not os.path.exists(model_file):
        raise Exception(f"File {model_file} not found")
    pipe = joblib.load(model_file)
    return pipe