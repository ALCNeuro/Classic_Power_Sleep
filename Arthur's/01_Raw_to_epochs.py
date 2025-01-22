#%%
import mne
import os
import pandas as pd
from glob import glob
import numpy as np
import scipy.io as sio  # For saving as .mat files
import matplotlib.pyplot as plt
from mne import Epochs
from mne.beamformer import make_lcmv, apply_lcmv_epochs

# Paths
raw_dir = '/Users/arthurlecoz/Desktop/boki'
data_dir = os.path.join(raw_dir, 'Raw')
preprocessed_dir = os.path.join(raw_dir, 'Preproc')
results_dir = os.path.join(raw_dir, 'Figs')

# Create necessary directories if they don't exist
# os.makedirs(raw_dir, exist_ok=True)
# os.makedirs(preprocessed_dir, exist_ok=True)
# os.makedirs(results_dir, exist_ok=True)

# Helper function to convert time to sample indices
def time_to_sample(raw, time_in_seconds):
    return int(time_in_seconds * raw.info['sfreq'])

files = glob(os.path.join(data_dir, "*.edf"))

#%% Loop over all subjects in the data directory

file = files[0]

for i_f, file in enumerate(files) : 
    
    sub_id = file.split('Raw/')[1].split('.edf')[0]
    
    scoring_savepath = os.path.join(data_dir, f"{sub_id}.txt")

    # Import the raw EEG data not loaded for memory allocation
    raw = mne.io.read_raw_edf(file, preload=False)
    
    # Define the montage
    coregistration_file = os.path.join(
        raw_dir, 'Co-registered_average_positions.pos'
        )
    montage_data = pd.read_csv(coregistration_file, sep='\t', header=None)
    electrode_names = montage_data[1].str.upper().tolist()
    coordinates = montage_data[[2, 3, 4]].values
    coordinates[:, [0, 1]] = coordinates[:, [1, 0]]  # Swap X and Y
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(electrode_names, coordinates)), coord_frame='head')
    
    # Set montage and clean data
    raw.rename_channels({ch_name: ch_name.replace('-Ref', '').upper() for ch_name in raw.ch_names if '-Ref' in ch_name})
    raw.set_montage(montage, on_missing='ignore')
    
    # Identify bad channels that are missing location info
    bad_channels = [ch['ch_name'] for ch in raw.info['chs'] if not np.isfinite(ch['loc'][:3]).all()]
    exclude_channels = [
        'ChEMG1', 'ChEMG2', 'RLEG-', 'RLEG+', 'LLEG-', 'LLEG+', 'EOG1', 
        'EOG2', 'ECG1', 'ECG2', 'SO1', 'SO2', 'ZY1', 'ZY2'
        ]
    # Exclude channels that are missing locations and non-EEG channels
    raw_eeg_cleaned = raw.copy().drop_channels(bad_channels + exclude_channels)
    half_night = raw_eeg_cleaned.copy().crop(tmin=0, tmax=len(raw_eeg_cleaned)/1000/2)

    # Load data before filtering
    half_raw = half_night.copy().load_data(verbose=None)
    raw_loaded = raw_eeg_cleaned.copy().load_data(verbose=None)
    
    # Filtering and resampling
    raw_filtered = half_raw.notch_filter(freqs=[50, 100], method='spectrum_fit', n_jobs=-1)  # Notch filter 50 and 100 Hz
    raw_filtered.filter(1, 45, fir_design='firwin', n_jobs = -1)  # Band-pass filter between 1-45 Hz
    raw_filtered.resample(256, npad='auto', n_jobs = -1)  # Resample to 256 Hz

    # Load the sleep stage annotations
    sleep_stages_df = pd.read_csv(scoring_savepath, sep='\t', header=None)
    sleep_stages_df.columns = ['label', 'start_time', 'duration']
    sleep_stages_mapping = {'W': 1, 'N1': 2, 'N2': 3, 'N3': 4, 'R': 5}

    # Apply average reference to the cleaned EEG data and apply projection
    raw_filtered.set_eeg_reference('average', projection=True)
    raw_filtered.apply_proj()

    # Add annotations to the raw
    scoring_annotations = mne.Annotations(
        onset = sleep_stages_df.start_time.values,
        duration = sleep_stages_df.duration.values,
        description = sleep_stages_df.label.values)
    raw_filtered.set_annotations(scoring_annotations)
    
    events, event_id = mne.events_from_annotations(raw_filtered)
    
    epochs=mne.Epochs(
        raw_filtered, 
        events=events,
        event_id=event_id,
        tmin=0, 
        tmax=30,
        baseline=None,
        preload=True
        )
    
    this_epochs_savename=os.path.join(preprocessed_dir, f"{sub_id}_epo.fif")
    epochs.save(this_epochs_savename)
    

    



