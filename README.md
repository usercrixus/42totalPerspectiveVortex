# MyBCI

This project is a simple Brain-Computer Interface (BCI) pipeline using the **EEG Motor Movement/Imagery Dataset** from PhysioNet.  
It extracts EEG epochs, filters them, and trains a **CSP + LDA classifier** to recognize motor imagery tasks (left vs right hand, fists vs feet).

---

## Requirements

- Python 3.9+
- Dependencies: `numpy`, `scipy`, `scikit-learn`, `mne`, `matplotlib`, `joblib`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the EEG dataset (109 subjects, 14 runs each) directly from PhysioNet:

```bash
make get_data
```

This will fetch files into `physionet.org/files/eegmmidb/1.0.0/`.

---

## Usage

The project is controlled via the **Makefile**.  
But first : python3 -m venv venv && source venv/bin/activate && pip install -r requirement

### Train a model

```bash
make train
```

(Example runs subject `10`, run `10`)

### Predict with a model

```bash
make predict
```

Runs predictions on the same subject/run and prints accuracy.

### Visualize filtering

```bash
make vizualize
```

Shows EEG before/after filtering and compares PSD.

### Run all experiments

```bash
make
```

Trains and evaluates models across subjects and runs.  
**Global mean accuracy: 0.635**

---

## Notes

- Models are stored in the `models/` folder.  
- Data must be downloaded before training/predicting.  
- The dataset is from [PhysioNet EEG Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/).  

---