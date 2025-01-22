#%%
import mne
import os
import pickle
from glob import glob
import numpy as np
import multiprocessing

# Paths
raw_dir = '/Users/arthurlecoz/Desktop/boki'
preprocessed_dir = os.path.join(raw_dir, 'Preproc')
power_dir = os.path.join(raw_dir, 'Power')

# Create necessary directories if they don't exist
# os.makedirs(raw_dir, exist_ok=True)
# os.makedirs(preprocessed_dir, exist_ok=True)
# os.makedirs(results_dir, exist_ok=True)

sfreq = 256

freqs = np.linspace(1, 40, 157)

method = "welch"
fmin = 1
fmax = 40
n_fft = 4*sfreq
n_per_seg = n_fft
n_overlap = int(n_per_seg/2)
window = "hamming"

threshold = dict(eeg = 300e-6)

files = glob(os.path.join(preprocessed_dir, "*.fif"))

stages = ["W", "N1", "N2", "N3", "R"]
channels = np.array(
    ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
       'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'F9', 'F10', 'T9',
       'T10', 'P9', 'P10', 'AF7', 'AF3', 'F11', 'F5', 'F1', 'FT11', 'FT9',
       'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'C5', 'C1', 'TP11', 'TP9',
       'TP7', 'CP3', 'CP1', 'P11', 'P5', 'P1', 'PO7', 'PO3', 'POZ', 'OZ',
       'FPZ', 'AFZ', 'AF4', 'AF8', 'F2', 'F6', 'F12', 'FC2', 'FC4', 'FC6',
       'FT8', 'FT10', 'FT12', 'C6', 'C2', 'CPZ', 'CP2', 'CP4', 'CP6',
       'TP8', 'TP10', 'TP12', 'P2', 'P6', 'P12', 'PO4', 'PO8', 'CP5']
    )

#%% Loop over all subjects in the data directory

file = files[0]

def compute_periodic_psd(file) :
    
    # if subtype.startswith('N') : continue
    sub_id = file.split('Preproc/')[-1].split('_epo')[0]
    # session = sub_id[-2:]
    
    this_subject_savepath = os.path.join(
        power_dir, f"{sub_id}_global_power.pickle"
        )
    
    if not os.path.exists(this_subject_savepath) : 
    
        temp_dic = {stage : {chan : [] for chan in channels}
                              for stage in stages}

        print(f"...processing {sub_id}")
        
        epochs = mne.read_epochs(file, preload = True)
        epochs.drop_bad(threshold)
        # sf = epochs.info['sfreq']
        
        # metadata = epochs.metadata
        
        for st in stages:
            print(f'processing {st}')
            if st not in epochs.event_id.keys() : 
                for channel in channels :
                    temp_dic[st][channel].append(np.nan*np.empty(freqs.shape[0]))
            else : 
                temp_list = []
                temp_power = epochs[st].compute_psd(
                        method = method,
                        fmin = fmin, 
                        fmax = fmax,
                        n_fft = n_fft,
                        n_overlap = n_overlap,
                        n_per_seg = n_per_seg,
                        window = window,
                        picks = channels
                        )
                for i_ch, channel in enumerate(channels) :
                    print(f'processing channel {channel}')
                    for i_epoch in range(len(epochs[st])) :
                        this_power = temp_power[i_epoch]
                        # psd = lowess(np.squeeze(
                        #     this_power.copy().pick(channel).get_data()), 
                        #     freqs, 0.075)[:, 1]
                        psd = np.squeeze(
                            this_power.copy().pick(channel).get_data())
                        
                        if np.any(psd < 0) :
                            for id_0 in np.where(psd<0)[0] :
                                psd[id_0] = abs(psd).min()
                                
                        temp_list.append(psd) 
                        
                    temp_dic[st][channel].append(
                        np.nanmean(temp_list, axis = 0)
                        )
                    
        with open (this_subject_savepath, 'wb') as handle:
            pickle.dump(temp_dic, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    from glob import glob
    # Get the list of EEG files
    eeg_files = files
    
    # Set up a pool of worker processes
    pool = multiprocessing.Pool(processes = 4)
    
    # Process the EEG files in parallel
    pool.map(compute_periodic_psd, eeg_files)
    
    # Clean up the pool of worker processes
    pool.close()
    pool.join()
