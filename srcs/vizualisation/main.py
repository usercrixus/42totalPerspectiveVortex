import mne
import matplotlib.pyplot as plt

def visualize_raw(file_path):
    """
    Load and plot raw EEG data from an EDF file.
    """
    print(f"Loading raw EEG from {file_path}...")
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.plot(n_channels=10, duration=10.0, title="Raw EEG (unfiltered)")
    plt.show(block=True)


def visualize_filtered(file_path, l_freq, h_freq, picks=None):
    print(f"Loading and filtering {file_path}...")
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)

    if picks:
        # Clean up channel names in the file to match standard names
        available = {ch.replace(".", "").replace(" ", ""): ch for ch in raw_filtered.ch_names}
        selected = [available[ch] for ch in picks if ch in available]

        print(f"Requested: {picks}")
        print(f"Available (normalized): {list(available.keys())}")
        print(f"Selected (actual names in file): {selected}")

        if not selected:
            print(f"⚠️ No requested channels found in this file.")
        else:
            raw_filtered.pick_channels(selected)
            raw_filtered.reorder_channels(selected)

    raw_filtered.plot(n_channels=10, duration=10.0, title="Filtered EEG (bandpass)")
    raw_filtered.compute_psd(fmax=50).plot(average=True)
    plt.show(block=True)


if __name__ == "__main__":
    file = "DATA/files/S001/S001R05.edf"
    '''
    What we can see is that some electrodes seem more relevant
    Fc4 Fc2 seem a good representation of the top
    Af7 Fp2 seem to have nice peaks
    T8 T10 Tp8 seem very high frequency
    '''
    visualize_raw(file)
    selected_channel = ['Fc4', 'Fc2', 'Af7', 'Fp2', 'T8', 'T10', 'Tp8']
    l_freq = 8.
    h_freq = 35.
    visualize_filtered(file, l_freq, h_freq, selected_channel)
