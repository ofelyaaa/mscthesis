import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
from scipy.signal import welch
from scipy.stats import ttest_rel

# =============================================================================
# === 1. CONFIGURATION & SETUP ===
# =============================================================================

# --- Paths and Files ---
base_data_path = ''
output_path = ''
montage_path = 'eeglab2024.0/sample_locs/Standard-10-20-Cap81.locs'
os.makedirs(output_path, exist_ok=True)

# --- Caching Configuration ---
# Set USE_CACHE to False or delete the cache_file to force re-processing.
USE_CACHE = True
cache_file = os.path.join(output_path, 'bandpower_results_full.csv')

# --- Participant Info ---
P_NUMBERS = [2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
GROUP_MAPPING = {
    2: 'non-ADHD', 3: 'non-ADHD', 4: 'ADHD', 5: 'non-ADHD', 7: 'non-ADHD', 8: 'non-ADHD',
    9: 'ADHD', 11: 'non-ADHD', 12: 'non-ADHD', 13: 'ADHD', 14: 'non-ADHD', 15: 'ADHD',
    16: 'ADHD', 17: 'ADHD', 18: 'ADHD', 19: 'ADHD', 20: 'ADHD', 21: 'non-ADHD',
    22: 'ADHD', 23: 'ADHD', 24: 'non-ADHD',
}
# --- Artifact & Channel Exclusion ---
ARTIFACT_ANNOTATION_LABELS = ['move']

# --- Event/Segmenting Parameters ---
stim_channel = 'Status'
first_stim_code = 5
bookmark_code = 3

# --- Analysis Parameters ---
FREQ_BANDS = {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 90]}
BANDS_ORDER = list(FREQ_BANDS.keys())
N_CONDITIONS = 6

# --- Configuration for Exploratory Analysis ---
RUN_EXPLORATORY_ANALYSIS = True
SPECTRAL_PROFILE_CHANNELS = ['CP2', 'CP6', 'C4']

# The four core experimental conditions of your factorial design
TASK_CONDITIONS = ['highintensity_highreverb', 'highintensity_lowreverb', 'lowintensity_highreverb',
                   'lowintensity_lowreverb']


# =============================================================================
# === 2. HELPER FUNCTIONS ===
# =============================================================================
def extract_condition_name(wav_path):
    base = os.path.basename(str(wav_path))
    name = base.replace('.wav', '').replace(' ', '').replace(',', '_').lower()
    if name == 'silent2': return 'silence'
    return name


def get_condition_labels(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if 43 >= df.shape[1]: raise ValueError(f"CSV file {csv_path} has no column AR (index 43).")
    wav_paths = df[43].dropna().astype(str)
    wav_paths = wav_paths[wav_paths.str.endswith('.wav')]
    cleaned = wav_paths.apply(extract_condition_name).reset_index(drop=True)
    return cleaned.groupby((cleaned != cleaned.shift()).cumsum()).first().tolist()


def get_condition_segments(raw, condition_labels):
    events = mne.find_events(raw, stim_channel=stim_channel, initial_event=True)
    five_events = events[events[:, 2] == first_stim_code]
    three_events = events[events[:, 2] == bookmark_code]
    if not len(five_events) or not len(three_events): return None, None, None
    trials_per_condition = len(five_events) // len(condition_labels)
    starts = [five_events[i * trials_per_condition, 0] for i in range(len(condition_labels))]
    ends = []
    for s in starts:
        following_threes = three_events[three_events[:, 0] > s]
        ends.append(following_threes[0, 0] - 1 if len(following_threes) > 0 else raw.n_times - 1)
    if len(starts) != len(condition_labels): return None, None, None
    return starts, ends, events


def calculate_band_power(data, sfreq, bands):
    freqs, psd = welch(data, sfreq, nperseg=int(2 * sfreq))
    return {band: np.sum(psd[:, np.logical_and(freqs >= low, freqs <= high)], axis=1) for band, (low, high) in
            bands.items()}


# =============================================================================
# === 3. DATA LOADING & PROCESSING ===
# =============================================================================
print(f"Loading custom montage from: {montage_path}")
try:
    custom_montage = mne.channels.read_custom_montage(fname=montage_path)
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load montage file. Error: {e}")
    exit()

info_template = None
results_df = None

# Check if we should use the cache
if USE_CACHE and os.path.exists(cache_file):
    print(f"\n--- Loading data from cache file: {cache_file} ---")
    results_df = pd.read_csv(cache_file)
    try:
        print("Loading template 'info' object for plotting...")
        first_p_fif = os.path.join(base_data_path, f'Participant{P_NUMBERS[0]}_cleaned_raw.fif')
        info_template = mne.io.read_raw_fif(first_p_fif, preload=False).pick_types(eeg=True).info
        info_template.set_montage(custom_montage, on_missing='warn')
    except Exception as e:
        print(f"❌ WARNING: Could not load a template 'info' object for plotting. Topomaps may fail. Error: {e}")

if results_df is None:
    print("\n--- Cache not found or not used. Starting full data processing... ---")
    all_results = []
    for p_num in P_NUMBERS:
        print(f"\n{'=' * 20} Processing Participant {p_num} {'=' * 20}")
        try:
            fif_path = os.path.join(base_data_path, f'Participant{p_num}_cleaned_raw.fif')
            csv_path = os.path.join(base_data_path, f'VisualResult_{p_num}.csv.csv')
            raw = mne.io.read_raw_fif(fif_path, preload=True)
            sfreq = raw.info['sfreq']
            raw.set_montage(custom_montage, on_missing='warn')
            artifact_indices = np.zeros(raw.n_times, dtype=bool)
            for ann in raw.annotations:
                if ann['description'] in ARTIFACT_ANNOTATION_LABELS:
                    start_sample, stop_sample = int(ann['onset'] * sfreq), int((ann['onset'] + ann['duration']) * sfreq)
                    artifact_indices[start_sample:stop_sample] = True
            if np.any(artifact_indices): raw._data[:, artifact_indices] = 0
            condition_labels = get_condition_labels(csv_path)
            starts, ends, _ = get_condition_segments(raw, condition_labels)
            if starts is None: continue
            raw.pick_types(eeg=True)
            if info_template is None: info_template = raw.info
            first_exp_start_sec = starts[0] / sfreq
            rest_data = None
            if first_exp_start_sec >= 100:
                rest_data = raw.copy().crop(tmin=first_exp_start_sec - 100, tmax=first_exp_start_sec).get_data()
            else:
                gap_data = [raw.copy().crop(tmin=(ends[j] / sfreq) + (1 / sfreq), tmax=starts[j + 1] / sfreq).get_data()
                            for
                            j in range(len(ends) - 1) if starts[j + 1] > ends[j]]
                if gap_data: rest_data = np.concatenate(gap_data, axis=1)
            if rest_data is not None and rest_data.shape[1] > sfreq:
                rest_power = calculate_band_power(rest_data, sfreq, FREQ_BANDS)
                total_rest_power = np.sum(list(rest_power.values()), axis=0)
                for ch_idx, ch_name in enumerate(raw.ch_names):
                    row = {'participant': p_num, 'group': GROUP_MAPPING.get(p_num), 'condition': 'rest',
                           'channel': ch_name}
                    for band in BANDS_ORDER: row[band] = rest_power[band][ch_idx]
                    row['total_power'] = total_rest_power[ch_idx]
                    all_results.append(row)
            for j, cond_name in enumerate(condition_labels):
                exp_data = raw.copy().crop(tmin=starts[j] / sfreq, tmax=ends[j] / sfreq).get_data()
                if exp_data.shape[1] > sfreq:
                    exp_power = calculate_band_power(exp_data, sfreq, FREQ_BANDS)
                    total_exp_power = np.sum(list(exp_power.values()), axis=0)
                    for ch_idx, ch_name in enumerate(raw.ch_names):
                        row = {'participant': p_num, 'group': GROUP_MAPPING.get(p_num), 'condition': cond_name,
                               'channel': ch_name}
                        for band in BANDS_ORDER: row[band] = exp_power[band][ch_idx]
                        row['total_power'] = total_exp_power[ch_idx]
                        all_results.append(row)
        except Exception as e:
            print(f"❌ FAILED to process P{p_num}: {e}")

    # =============================================================================
    # === 4. DATA AGGREGATION & POST-PROCESSING ===
    # =============================================================================
    print("\n--- Data Processing Complete ---")
    if not all_results:
        print("❌ No data was processed. Exiting.")
        exit()
    results_df = pd.DataFrame(all_results)

    epsilon = 1e-12
    for band in BANDS_ORDER: results_df[f'{band}_norm'] = results_df[band] / (results_df['total_power'] + epsilon)

    results_df['alpha_theta_ratio'] = results_df['alpha'] / (results_df['theta'] + epsilon)
    results_df['theta_beta_ratio'] = results_df['theta'] / (results_df['beta'] + epsilon)
    results_df['beta_alpha_ratio'] = results_df['beta'] / (results_df['alpha'] + epsilon)
    results_df['gamma_theta_ratio'] = results_df['gamma'] / (results_df['theta'] + epsilon)
    results_df['engagement_ratio'] = results_df['beta'] / (results_df['alpha'] + results_df['theta'] + epsilon)

    results_df.to_csv(cache_file, index=False)
    print(f"Full results saved to cache file: {cache_file}")

# =============================================================================
# === 7. EXPLORATORY VISUAL ANALYSIS ===
# =============================================================================
if RUN_EXPLORATORY_ANALYSIS and not results_df.empty:
    print("\n--- Running Exploratory Visual Analysis (Plots will be saved to file) ---")
    plot_order = TASK_CONDITIONS + ['white_noise', 'silence', 'rest']

    # --- 1. Spectral Profile Bar Plots ---
    for channel in SPECTRAL_PROFILE_CHANNELS:
        fig, ax = plt.subplots(figsize=(15, 7)) # Use fig, ax for clarity
        channel_df = results_df[results_df['channel'] == channel].copy()
        id_vars = ['group', 'condition']
        value_vars = [f'{b}_norm' for b in BANDS_ORDER]
        melted_df = channel_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='band', value_name='power')
        melted_df['band'] = melted_df['band'].str.replace('_norm', '')
        sns.barplot(data=melted_df, x='condition', y='power', hue='band', order=plot_order, hue_order=BANDS_ORDER,
                    palette='viridis', ax=ax)
        ax.set_title(f'Normalized Spectral Profile at Channel: {channel}', fontsize=16)
        ax.set_ylabel('Normalized Power')
        ax.set_xlabel('Condition')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Band', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'exploratory_spectral_profile_{channel}.png'), dpi=300)

    # --- 2. Factorial Contrast Topomaps ---
    if info_template is None:
        print("❌ WARNING: Cannot generate topomaps because no 'info' object is available.")
    else:
        def get_mean_map(df, cond_list, band='alpha_norm'):
            return df[df['condition'].isin(cond_list)].groupby('channel')[band].mean()
        def get_mean_ratio(df, cond_list, ratio_col='alpha_theta_ratio'):
            subset = df[df['condition'].isin(cond_list)]
            return subset[ratio_col].mean() if not subset.empty else np.nan

        high_intensity_map = get_mean_map(results_df, ['highintensity_highreverb', 'highintensity_lowreverb'])
        low_intensity_map = get_mean_map(results_df, ['lowintensity_highreverb', 'lowintensity_lowreverb'])
        all_task_map = get_mean_map(results_df, TASK_CONDITIONS)
        silence_map = get_mean_map(results_df, ['silence'])
        contrasts = {"Intensity (High-Low)": high_intensity_map - low_intensity_map, "Task - Silence": all_task_map - silence_map}
        ratios_to_display = {
            "Intensity (High-Low)": (f"Mean A/T Ratio\n" f"High: {get_mean_ratio(results_df, ['highintensity_highreverb', 'highintensity_lowreverb']):.2f} | " f"Low: {get_mean_ratio(results_df, ['lowintensity_highreverb', 'lowintensity_lowreverb']):.2f}"),
            "Task - Silence": (f"Mean A/T Ratio\n" f"Task: {get_mean_ratio(results_df, TASK_CONDITIONS):.2f} | " f"Silence: {get_mean_ratio(results_df, ['silence']):.2f}")
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Topographical Contrasts for Normalized Alpha Power', fontsize=20)
        im = None
        for ax, (title, data) in zip(axes, contrasts.items()):
            if not data.empty:
                max_abs = np.abs(data).max()
                im, _ = mne.viz.plot_topomap(data, info_template, axes=ax, show=False, cmap='RdBu_r', vlim=(-max_abs, max_abs))
                ax.set_title(title, fontsize=14)
                ax.set_xlabel(ratios_to_display.get(title, ""), fontsize=11, labelpad=10)
            else:
                ax.set_title(f'{title}\n(No data)', fontsize=14)
                ax.axis('off')
        fig.tight_layout(rect=[0, 0.05, 0.9, 0.93])
        if im:
            cbar_ax = fig.add_axes([0.92, 0.2, 0.03, 0.6])
            fig.colorbar(im, cax=cbar_ax, label='Difference in Norm. Alpha Power')
        plt.savefig(os.path.join(output_path, 'exploratory_factorial_contrasts.png'), dpi=300)
        # plt.show() # MODIFICATION: Removed to show all at the end

    # --- 3. Alpha-Theta Ratio Topomaps by Group and Condition (FINAL STYLE) ---
    if info_template is None:
        print("❌ WARNING: Cannot generate Alpha/Theta ratio topomaps because no 'info' object is available.")
    else:
        print("\n--- Generating Final V2 Mean Alpha/Theta Ratio Topomaps ---")

        # 1. Update the mapping for the new condition title format
        CONDITION_NAME_MAP = {
            'highintensity_highreverb': 'High intensity, High reverb.',
            'highintensity_lowreverb': 'High intensity, Low reverb.',
            'lowintensity_highreverb': 'Low intensity, High reverb.',
            'lowintensity_lowreverb': 'Low intensity, Low reverb.',
            'white_noise': 'White noise.',
            'silence': 'Silence',
            'rest': 'Rest'
        }
        conditions_to_plot = list(CONDITION_NAME_MAP.keys())

        plot_df = results_df[results_df['condition'].isin(conditions_to_plot)].copy()
        plot_df['condition'] = pd.Categorical(plot_df['condition'], categories=conditions_to_plot, ordered=True)

        # 2. Revert sort order to place the 'ADHD' group on the top row
        groups = sorted(plot_df['group'].dropna().unique())
        n_groups = len(groups)
        n_conditions = len(conditions_to_plot)

        all_mean_ratios = plot_df.groupby(['group', 'condition', 'channel'])['alpha_theta_ratio'].mean()
        vmin = all_mean_ratios.min()
        vmax = all_mean_ratios.max()

        fig, axes = plt.subplots(n_groups, n_conditions,
                                 figsize=(20, 8),  # Adjusted figsize for new labels
                                 gridspec_kw=dict(hspace=0.3, wspace=0.1))

        im = None

        for i, group in enumerate(groups):
            for j, condition in enumerate(conditions_to_plot):
                ax = axes[i, j]

                if j == 0:
                    ax.set_ylabel(group, fontsize=20, labelpad=40, rotation=0, ha='right', va='center')

                subset_df = plot_df[(plot_df['group'] == group) & (plot_df['condition'] == condition)]

                if not subset_df.empty:
                    topoplot_data = subset_df.groupby('channel')['alpha_theta_ratio'].mean()
                    if not topoplot_data.empty:
                        im, _ = mne.viz.plot_topomap(topoplot_data, info_template,
                                                     axes=ax, show=False,
                                                     cmap='viridis', vlim=(vmin, vmax))

                        # 3. Update statistic format to Mean(SD) without labels
                        mean_val = topoplot_data.mean()
                        std_val = topoplot_data.std()
                        label_text = f"{mean_val:.2f}({std_val:.2f})"
                        ax.set_xlabel(label_text, fontsize=12, labelpad=10)

                    else:
                        ax.axis('off')
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

                # 4. Use new names and rotate titles to prevent overlap
                if i == 0:
                    clear_title = CONDITION_NAME_MAP.get(condition, condition)
                    ax.set_title(clear_title, fontsize=12, rotation=35, ha='left', va='bottom')

        # Adjust layout to ensure rotated titles and labels fit
        fig.tight_layout(rect=[0.03, 0.05, 0.92, 0.90])
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        if im:
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Alpha/Theta Power Ratio', size=14)

        final_plot_path_v2 = os.path.join(output_path, 'topoplot_alpha_theta_ratio_final_v2.png')
        plt.savefig(final_plot_path_v2, dpi=300, bbox_inches='tight')
        print(f"--- Final V2 plot saved to: {final_plot_path_v2} ---")

        plt.show()

# =============================================================================
# === 8. EXPLORATORY NUMERICAL RESULTS ===
# =============================================================================
if RUN_EXPLORATORY_ANALYSIS and not results_df.empty:
    print("\n" + "=" * 80)
    print("--- EXPLORATORY NUMERICAL RESULTS ---")
    print("=" * 80)

    print("\n[Summary of Alpha Power Contrasts]\n")
    for title, data in contrasts.items():
        if not data.empty:
            max_ch = data.idxmax()
            min_ch = data.idxmin()
            print(f"For '{title}':")
            print(f"  - Largest Increase at channel '{max_ch}' (Value: {data.max():.4f})")
            print(f"  - Largest Decrease at channel '{min_ch}' (Value: {data.min():.4f})\n")

    print("\n[Summary of Group Differences in Task Conditions]\n")
    task_df = results_df[results_df['condition'].isin(TASK_CONDITIONS)]
    group_engagement = task_df.groupby('group')['engagement_ratio'].mean()
    if 'ADHD' in group_engagement.index and 'nonADHD' in group_engagement.index:
        print("Mean Engagement Ratio (Beta / (Alpha+Theta)) during all task conditions:")
        print(f"  - ADHD Group:    {group_engagement['ADHD']:.4f}")
        print(f"  - nonADHD Group: {group_engagement['nonADHD']:.4f}")
        diff = group_engagement['ADHD'] - group_engagement['nonADHD']
        print(f"  - Difference (ADHD - nonADHD): {diff:.4f}\n")

    print("\n[Summary of Peak Power Conditions per Group]\n")
    for group in results_df['group'].dropna().unique():
        print(f"For '{group}' group:")
        group_df = results_df[results_df['group'] == group]
        peak_alpha_cond = group_df.groupby('condition')['alpha_norm'].mean().idxmax()
        peak_alpha_val = group_df.groupby('condition')['alpha_norm'].mean().max()
        print(f"  - Peak Normalized Alpha Power occurred during '{peak_alpha_cond}' (Value: {peak_alpha_val:.4f})")
        peak_gamma_cond = group_df.groupby('condition')['gamma_norm'].mean().idxmax()
        peak_gamma_val = group_df.groupby('condition')['gamma_norm'].mean().max()
        print(f"  - Peak Normalized Gamma Power occurred during '{peak_gamma_cond}' (Value: {peak_gamma_val:.4f})\n")

print("\n--- Analysis Pipeline Finished ---")
