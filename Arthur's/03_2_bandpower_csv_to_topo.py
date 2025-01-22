#%%
import mne
import os
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths
raw_dir = '/Users/arthurlecoz/Desktop/boki'
preprocessed_dir = os.path.join(raw_dir, 'Preproc')
power_dir = os.path.join(raw_dir, 'Power')

# Create necessary directories if they don't exist
# os.makedirs(raw_dir, exist_ok=True)
# os.makedirs(preprocessed_dir, exist_ok=True)
# os.makedirs(results_dir, exist_ok=True)

bp_files = glob(os.path.join(power_dir, "*bandpower.csv"))

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

midline = ["FZ", "CZ", "PZ", "OZ"]

epochs = mne.read_epochs(
    glob(os.path.join(preprocessed_dir, "*epo.fif"))[0], 
         preload = True)

cols_power = [
    'abs_delta','abs_theta','abs_alpha','abs_beta','abs_gamma',
    'rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma'
    ]

#%% Gather all of the dataframe and concat it

all_subject_bp_savename = os.path.join(
    power_dir, "all_subject_bandpower.csv"
    )

if not os.path.exists(all_subject_bp_savename) :
    all_csv = [pd.read_csv(csv) for csv in bp_files]
    df = pd.concat(all_csv)
    del df['Unnamed: 0']
else : df = pd.read_csv(all_subject_bp_savename)

# %% Group it by subject however you want 

"""
You obtained a dataframe for each subject for each epochs.
You want to average per subject to have a value of absolute and relative power
per frequency band per sleep stage.

But depending on how you want to plot it afterward, you can adjust it.

Here I'll code it so you can compare everything to wake, but you could play
with it.
"""

mean_df = df.groupby(['sub_id', 'channel', 'stage'], as_index = False).mean()

# %% Topographies all sleep stages : Eye Check

info = epochs.info
channel_order = np.array(epochs.ch_names)

vlims = 1

for feature in cols_power :
    fig, ax = plt.subplots(
        nrows = 1, ncols = len(stages), figsize = (16, 4), layout = 'tight')
    
    columns = [feature, "channel"]
    
    if vlims :
        vmin = df[columns].groupby(
            'channel', as_index = False).mean().min()[feature]
        vmax = df[columns].groupby(
            'channel', as_index = False).mean().max()[feature]
    
    for i_st, stage in enumerate(stages) :
        
        temp_values = df[columns].loc[
            (df.stage == stage)
            ]
        
        temp_values = temp_values.groupby('channel', as_index = False).mean()
        temp_values['channel'] = pd.Categorical(
            temp_values['channel'], 
            categories=channel_order, 
            ordered=True
            )
        df_sorted = temp_values.sort_values('channel')

        values = df_sorted[feature].to_numpy()
    
        divider = make_axes_locatable(ax[i_st],)
        cax = divider.append_axes("right", size = "5%", pad=0.05)
        im, cm = mne.viz.plot_topomap(
            data = values,
            pos = info,
            axes = ax[i_st],
            contours = 2,
            cmap = "Purples",
            vlim = (vmin, vmax)
            )
        fig.colorbar(im, cax = cax, orientation = 'vertical')
        ax[i_st].set_title(f"{stage}", fontweight = "bold")
    
    fig.suptitle(f"{feature}\n", 
             fontsize = "xx-large", 
             fontweight = "bold")   
    
    plt.show()  

# %% Topographies : Statistical comparisons

interest = 'abs_beta'
fdr_corrected = 0

model = f"{interest} ~ C(stage, Treatment('W'))" 

fig, ax = plt.subplots(
    nrows = 1, ncols = len(stages)-1, figsize = (14, 4))

for i, stage in enumerate(stages[1:]):
    
    temp_tval = []; temp_pval = []; chan_l = []
    cond_df = mean_df.loc[mean_df.stage.isin(['W', stage])]
    for chan in channels :
        subdf = cond_df[
            ['sub_id', 'stage', 'channel', f'{interest}']
            ].loc[(cond_df.channel == chan)].dropna()
        md = smf.mixedlm(model, subdf, groups = subdf['sub_id'], missing = 'omit')
        mdf = md.fit()
        temp_tval.append(mdf.tvalues[f"C(stage, Treatment('W'))[T.{stage}]"])
        temp_pval.append(mdf.pvalues[f"C(stage, Treatment('W'))[T.{stage}]"])
        chan_l.append(chan)
        
    if np.any(np.isnan(temp_tval)) :
        temp_tval[np.where(np.isnan(temp_tval))[0][0]] = np.nanmean(temp_tval)
         
    if fdr_corrected :
        _, corrected_pval = fdrcorrection(temp_pval)
        display_pval = corrected_pval
    else : 
        display_pval = temp_pval
    
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes("right", size = "5%", pad=0.05)
    im, cm = mne.viz.plot_topomap(
        data = temp_tval,
        pos = epochs.info,
        axes = ax[i],
        contours = 3,
        mask = np.asarray(display_pval) <= 0.05,
        mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=6),
        cmap = "viridis",
        vlim = (np.percentile(temp_tval, 5), np.percentile(temp_tval, 95))
        )
    fig.colorbar(im, cax = cax, orientation = 'vertical')

    ax[i].set_title(f"{stage} > W", fontsize=12)
fig.suptitle(f"{interest}", fontsize=16)
fig.tight_layout(pad = 2)
