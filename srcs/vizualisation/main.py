# srcs/vizualisation/main.py

import mne
import matplotlib.pyplot as plt

# Use relative imports inside the package
from ..trainAndPredict.edf import getGlobalFilter, applyGlobalFilter
from ..trainAndPredict.utils import getRawEDF  # picks motor channels via MOTOR_LABELS


def _mean_psd(psd):
    """
    Return a 1D mean PSD over channels (and epochs if present).
    Handles shapes from Raw (C,F) and Epochs (E,C,F).
    """
    data = psd.get_data()
    if data.ndim == 3:      # (epochs, channels, freqs)
        return data.mean(axis=(0, 1))
    if data.ndim == 2:      # (channels, freqs)
        return data.mean(axis=0)
    raise ValueError(f"Unexpected PSD shape: {data.shape}")


def compare_psd(raw, raw_filt, fmax=50):
    """
    Overlay average PSD of raw vs filtered on one plot.
    """
    psd_raw = raw.compute_psd(fmax=fmax, verbose=False)
    psd_fil = raw_filt.compute_psd(fmax=fmax, verbose=False)

    freqs = psd_raw.freqs
    p_raw = _mean_psd(psd_raw)
    p_fil = _mean_psd(psd_fil)

    fig, ax = plt.subplots()
    ax.plot(freqs, p_raw, label="raw")
    ax.plot(freqs, p_fil, label="filtered")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("PSD before vs after filtering")
    ax.legend()
    plt.show(block=True)


def visualize_filtered(file_path: str):
    """
    Load EDF, build T0-based equalizer, apply zero-phase FIR, and visualize.
    """
    print(f"Loading and filtering {file_path}...")

    # Use project loader to get the same motor channels as during training
    raw = getRawEDF(file_path)

    # Build equalizer from this file's T0 and apply (zero-phase FIR via filtfilt)
    h = getGlobalFilter(raw)
    raw_filt = applyGlobalFilter(raw, h)

    # Clear accidental bad-channel markings for fair comparison
    raw.info["bads"] = []
    raw_filt.info["bads"] = []

    # Plot filtered first (one window at a time to avoid confusion)
    raw_filt.plot(n_channels=10, duration=10.0, title="Filtered EEG (T0-equalized)")
    plt.show(block=True)

    # Then plot raw
    raw.plot(n_channels=10, duration=10.0, title="Raw EEG (unfiltered)")
    plt.show(block=True)

    # PSD comparison overlay
    compare_psd(raw, raw_filt, fmax=50)


if __name__ == "__main__":
    file = "physionet.org/files/eegmmidb/1.0.0/S001/S001R05.edf"
    visualize_filtered(file)
