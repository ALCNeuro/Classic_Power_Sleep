# EEG Band Power Analysis Pipeline

This repository provides a set of Python scripts for processing EEG data to compute power spectral density (PSD), bandpower metrics, and statistical comparisons across different sleep stages. It leverages the MNE-Python library and other scientific Python tools.

## Features
- **Epoch Processing:** Convert raw EEG data to epochs for analysis.
- **Global Power Computation:** Calculate PSD using Welch's method for each sleep stage and channel.
- **Bandpower Metrics:** Derive absolute and relative bandpower (delta, theta, alpha, beta, gamma) for each channel and sleep stage.
- **Topographic Mapping:** Visualize power distributions across EEG channels as topographic maps.
- **Statistical Analysis:** Perform mixed-effects modeling and FDR-corrected comparisons between sleep stages.

## Repository Structure
1. **`raw_to_epochs.py`:** Processes raw EEG files into epochs and prepares initial datasets for analysis.
2. **`epochs_to_globalpower.py`:** Computes global PSD for each epoch and organizes data by sleep stages and channels.
3. **`global_power_to_psds.py`:** Aggregates and averages PSDs across subjects, producing stage-specific power plots.
4. **`epochs_to_bandpower_csv.py`:** Calculates bandpower metrics (absolute and relative) and exports the results to CSV files.
5. **`bandpower_csv_to_topo.py`:** Generates topographic maps and conducts statistical comparisons between sleep stages.

## Requirements
- Python 3.8+
- Libraries: `mne`, `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`
- Recommended: Multiprocessing support for large datasets.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/eeg-bandpower-analysis.git
   cd eeg-bandpower-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare Data:**
   Place raw EEG `.fif` files in the `Preproc` directory under your designated `raw_dir` path.
2. **Run Scripts in Sequence:**
   - Convert raw data to epochs: `raw_to_epochs.py`
   - Compute global power: `epochs_to_globalpower.py`
   - Aggregate PSDs: `global_power_to_psds.py`
   - Calculate bandpower: `epochs_to_bandpower_csv.py`
   - Generate visualizations and statistics: `bandpower_csv_to_topo.py`

## Outputs
- **Bandpower Metrics:** Saved as CSV files in the `Power` directory.
- **Topographic Maps:** Plotted for each band and sleep stage.
- **PSD Plots:** Frequency domain power distributions for selected channels.
- **Statistical Results:** Sleep stage comparisons visualized through topographic significance maps.

## Customization
- **Frequency Bands:** Adjust `freq_bands` in `epochs_to_bandpower_csv.py`.
- **Channels and Stages:** Modify `channels` and `stages` arrays in each script.
- **Visualization:** Change colormap or layout options in plotting sections.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to update this file to fit your specific project details or dataset!
