import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import json
from mne_icalabel import label_components

plt.ion()

# CONFIG
p_num =  #participant number here
base_data_path = ''
locs_path = 'eeglab2024.0/sample_locs/Standard-10-20-Cap81.locs'
data_path = os.path.join(base_data_path, f'Participant{p_num}.bdf')
exclude_file = os.path.join(base_data_path, f'Participant{p_num}_excluded_components.json')
annotations_fname = os.path.join(base_data_path, f'Participant{p_num}_annot.fif')

# 1. Load file and prepare
print(f"\nLoading BDF (with stim channel) from {data_path}")
raw = mne.io.read_raw_bdf(data_path, preload=True, stim_channel='Status')
events = mne.find_events(raw, stim_channel='Status', shortest_event=1, initial_event=True)

if events.size > 0:
    last_time = events[-1, 0] / raw.info['sfreq']
    raw.crop(tmax=last_time)
    print(f"Cropped raw data until last event at {last_time:.2f}s")

# 1. Load file and prepare
print(f"\nLoading BDF (with stim channel) from {data_path}")
raw = mne.io.read_raw_bdf(data_path, preload=True, stim_channel='Status')
events = mne.find_events(raw, stim_channel='Status', shortest_event=1, initial_event=True)

if events.size > 0:
    last_time = events[-1, 0] / raw.info['sfreq']
    raw.crop(tmax=last_time)
    print(f"Cropped raw data until last event at {last_time:.2f}s")
else:
    print("Warning: No events found. Not cropping data.")

# FOR P14 w/ wrong cfg file: identify bad channels
#bad_channels = ['LEOG', 'REOG', 'UEOG', 'DEOG', 'M1', 'M2', 'EXG7', 'EXG8']
# Drop them
#raw.drop_channels(bad_channels)

#  apply montage 
montage = mne.channels.read_custom_montage(locs_path)
raw.set_montage(montage, on_missing='warn')
raw.set_eeg_reference('average', projection=False)

# 2. FILTERING
raw_filt = raw.copy().filter(1., 100., fir_design='firwin').notch_filter(50., fir_design='firwin')

# CHECK FOR GROSS ARTIFACTS BEFORE ICA
if os.path.exists(annotations_fname):
    print(f"Loading previous annotations from {annotations_fname}")
    previous_annot = mne.read_annotations(annotations_fname)
    raw_filt.set_annotations(previous_annot)
else:
    print("No previous annotations found.")

if input("View and annotate uncleaned data before ICA? (y/n): ").lower() == 'y':
    raw_filt.plot(duration=10, n_channels=32, scalings=dict(eeg=100e-6),
                  title=f"Participant {p_num} – Uncleaned (Pre-ICA)", block=True, events=events)
    raw_filt.annotations.save(annotations_fname, overwrite=True)
    print(f"Annotations saved to {annotations_fname} after uncleaned inspection.")

if input("Mark bad channels before ICA? (y/n): ").lower() == 'y': #don't worry, ica is not run on the interpolated channels!
    print(f"Current channels: {raw_filt.ch_names}")
    bad_chs = input("Enter bad channel names separated by commas (e.g., 'Fz,Cz,Oz'): ").strip()
    if bad_chs:
        bad_chs_list = [ch.strip() for ch in bad_chs.split(',')]
        raw_filt.info['bads'].extend(bad_chs_list)
        raw_filt.info['bads'] = list(set(raw_filt.info['bads']))
        print(f"Marked bad channels: {raw_filt.info['bads']}")
        raw_filt.interpolate_bads(reset_bads=True)
        print("Interpolated bad channels.")

# 2b. DOWNSAMPLE FOR SPEEDY ICA
downsample_sfreq = 200
if raw_filt.info['sfreq'] > downsample_sfreq:
    raw_ica = raw_filt.copy().resample(downsample_sfreq, npad="auto")
else:
    raw_ica = raw_filt.copy()

# --- DIAGNOSTIC CHECKS ---
print("\n" + "="*20 + " DIAGNOSTICS " + "="*20)
print(f"Annotations on original filtered object (raw_filt):")
print(raw_filt.annotations)
print("-" * 50)
print(f"Annotations on downsampled object for ICA (raw_ica):")
print(raw_ica.annotations)
print("=" * 53 + "\n")

if input("Visually inspect the data being sent to ICA? (y/n): ").lower() == 'y':
    print("Plotting raw_ica. Look for your 'move' annotation regions.")
    raw_ica.plot(duration=15, n_channels=32, scalings=dict(eeg=100e-6), block=True)


# 3. FIT ICA
ica = mne.preprocessing.ICA(n_components=None, method='infomax', fit_params=dict(extended=True), random_state=97)
print("Fitting ICA…")
ica.fit(raw_ica, picks='eeg', reject_by_annotation='move')

# 3b. RUN ICLabel and DISPLAY FULL OUTPUT
print("Running ICLabel to classify ICA components…")
ic_labels = label_components(raw_ica, ica, method='iclabel')

labels = ic_labels['labels']
probs = ic_labels['y_pred_proba']

print("\n--- Full ICLabel Classification Output ---")
for i, (label, prob) in enumerate(zip(labels, probs)):
    print(f"Component {i:02d}: {label:<20} (p = {prob:.4f})")

# 4. AUTO EOG
frontal_chs = [ch for ch in ['Fp1', 'Fp2'] if ch in raw_ica.ch_names]
eog_inds = []
if frontal_chs:
    for ch in frontal_chs:
        inds, _ = ica.find_bads_eog(raw_ica, ch_name=ch, threshold='auto')
        eog_inds.extend(inds)
    eog_inds = sorted(set(eog_inds))


# 5. HEARTBEAT-LIKE BY LOW-FREQ PSD
def get_low_freq_power(ica, raw, band=(1, 4)):
    power = {}
    for i in range(ica.n_components_):
        ts = ica.get_sources(raw).get_data(picks=i).squeeze()
        psd, freqs = mne.time_frequency.psd_array_welch(ts, sfreq=raw.info['sfreq'], fmin=1, fmax=40, verbose=False)
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        power[i] = psd[band_mask].sum() / psd.sum()
    return power


low_freq_power = get_low_freq_power(ica, raw_ica)
heartbeat_candidates = [comp for comp, val in low_freq_power.items() if val > 0.8]

print("\n🧠 Summary of other component suggestions (for automatic inspection):")
print(f"EOG (find_bads_eog): {eog_inds}")
print(f"Heartbeat-like (low-freq PSD > 0.8): {heartbeat_candidates}")

all_candidates = sorted(set(eog_inds).union(heartbeat_candidates))

# 6. VISUALIZATION AND INSPECTION

# Plot 1: Bar chart of ICLabel results for a quick overview
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(ica.n_components_), probs, color='lightblue')
ax.set_ylabel('Probability')
ax.set_xlabel('Component Index')
ax.set_title(f'ICLabel Classification Confidence for Participant {p_num}')
# Add labels on top of bars
for i, (label, prob) in enumerate(zip(labels, probs)):
    ax.text(i, prob + 0.01, f"{label}\n({prob:.2f})", ha='center', va='bottom', fontsize=8, rotation=90)
ax.set_ylim(0, 1.1)
ax.set_xticks(range(ica.n_components_))
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show(block=True)

# Plot 2: All component topographies
print("\nPlotting all ICA component topographies...")
ica.plot_components(show=False)
plt.show(block=True)

# Plot 3: Interactive, detailed inspection loop
while True:
    print("\n" + "=" * 50)
    print("INTERACTIVE COMPONENT INSPECTION")
    print("Based on the ICLabel chart and topographies, enter component indices to inspect.")
    inp = input("Enter indices separated by commas (e.g., '0, 5, 12'), or press Enter to finish: ")
    print("=" * 50 + "\n")

    if not inp.strip():
        print("Exiting inspection loop.")
        break

    try:
        to_plot = [int(x.strip()) for x in inp.split(',')]
        for comp in to_plot:
            if 0 <= comp < ica.n_components_:
                print(f"\n--- Displaying detailed plots for Component {comp} ---")

                # Plot 3a: The standard properties plot
                ica.plot_properties(raw_ica, picks=[comp], psd_args={'fmax': 50.})
                plt.suptitle(f'Component {comp} - Properties', y=1.02)
                plt.show(block=True)

                # Plot 3b: The component time series itself
                ica.plot_sources(raw_ica, picks=[comp], title=f'Component {comp} - Time Series')
                plt.show(block=True)

                # Plot 3c: The overlay showing the effect of removal
                print(f"Plotting overlay for component {comp} removal...")
                ica.plot_overlay(raw_ica, exclude=[comp], picks='eeg', title=f'Effect of Removing Component {comp}')
                plt.show(block=True)

            else:
                print(f"Warning: Component {comp} is out of range (0-{ica.n_components_ - 1}). Skipping.")

    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
    except Exception as e:
        print("An error occurred during plotting:", e)

# 7. CHOOSE EXCLUSIONS
user_exclude = []
load_prev_choice = ''
if os.path.exists(exclude_file):
    load_prev_choice = input("Load previous exclusions from file? (y/n): ")

if load_prev_choice.lower() == 'y':
    with open(exclude_file, 'r') as f:
        user_exclude = json.load(f)
    print("Loaded exclusions:", user_exclude)
else:
    print("\n--- Review the full ICLabel list and plots to make your selection ---")
    inp = input("Enter ALL components to exclude, separated by commas: ")
    if inp.strip():
        user_exclude = [int(x.strip()) for x in inp.split(',')]
    else:
        user_exclude = []  # Default to excluding nothing if input is empty

    user_exclude = sorted(list(set(user_exclude)))
    with open(exclude_file, 'w') as f:
        json.dump(user_exclude, f)
    print(f"Saved exclusion list to {exclude_file}")

ica.exclude = user_exclude
print(f"Final exclusion list: {ica.exclude}")

# 8. APPLY TO FULL-FREQ DATA
raw_clean = raw_filt.copy()
ica.apply(raw_clean)

# 9. PLOT CLEANED DATA
if os.path.exists(annotations_fname):
    print(f"Loading previous annotations from {annotations_fname}")
    previous_annot = mne.read_annotations(annotations_fname)
    raw_clean.set_annotations(previous_annot)
else:
    print("No previous annotations found.")

if input("View uncleaned data first? (y/n): ").lower() == 'y':
    raw_filt.plot(duration=10, n_channels=32, scalings=dict(eeg=100e-6),
                  title=f"Participant {p_num} – Uncleaned", events=events, block=True)

print("Plotting cleaned data…")
raw_clean.plot(duration=10, n_channels=32, scalings=dict(eeg=100e-6),
               title=f"Participant {p_num} – Cleaned", events=events, block=True)

# 10. SAVE CLEANED FILES
if raw_clean.annotations and len(raw_clean.annotations) > 0:
    overwrite_annot = False
    if os.path.exists(annotations_fname):
        if input("Overwrite existing annotations file? (y/n): ").lower() == 'y':
            overwrite_annot = True
    else:
        overwrite_annot = True
    if overwrite_annot:
        raw_clean.annotations.save(annotations_fname, overwrite=True)
        print(f"Annotations saved to {annotations_fname}")

cleaned_fname = os.path.join(base_data_path, f'Participant{p_num}_cleaned_raw.fif')
raw_clean.save(cleaned_fname, overwrite=True)
print(f"\n✅ Preprocessing complete. Cleaned data saved to:\n{cleaned_fname}")
