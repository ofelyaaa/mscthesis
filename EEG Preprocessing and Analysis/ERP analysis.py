# -*- coding: utf-8 -*-
"""
Full EEG ERP Analysis Script (Corrected and Extended Version)

This script performs a complete ERP analysis pipeline:
1.  Loads pre-cleaned participant EEG data.
2.  Segments data by experimental condition and event type (stimulus/response).
3.  Creates epochs and calculates participant-level evoked responses (ERPs).
4.  Calculates and plots grand-average ERPs for different groups (ADHD vs. non-ADHD).
5.  Plots channel-specific comparisons (e.g., 'Oz').
6.  Plots Global Field Power (GFP) with individual participant overlays.
7.  Creates a summary image of topoplots at the peak GFP for all conditions and groups in a HORIZONTAL layout.
8.  Performs cluster-based permutation statistics to find group differences.
9.  Calculates peak latency and amplitude for specific channels and the GFP.
10. Saves all plots and statistical results to an output directory.

Caching is used to speed up re-runs of the analysis.
"""

# === 0. IMPORTS ===
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import ttest_ind_no_p

# === 1. CONFIGURATION ===
# --- Path Configuration ---
base_data_path = ''
output_path = ''
os.makedirs(output_path, exist_ok=True)

# --- Caching Configuration ---
USE_PROCESSED_DATA_CACHE = True
PROCESSED_DATA_CACHE = os.path.join(output_path, 'cache_participant_evokeds.pkl')
STATS_CACHE_DIR = os.path.join(output_path, 'cache_cluster_stats')
os.makedirs(STATS_CACHE_DIR, exist_ok=True)

# --- Participant & Group Configuration ---
P_NUMBERS = [2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
GROUP_MAPPING = {p: 'ADHD' if p in [4, 9, 13, 15, 16, 17, 18, 19, 20, 22, 23] else 'non-ADHD' for p in P_NUMBERS}
# MODIFICATION: 'rest' condition removed as it's not event-related
CONDITIONS = [
    'highintensity_highreverb', 'highintensity_lowreverb', 'lowintensity_highreverb',
    'lowintensity_lowreverb', 'silence', 'white_noise'
]
GROUPS = ['non-ADHD', 'ADHD']  # Define order for plotting

# --- ERP & Epoching Configuration ---
stim_channel = 'Status'
first_stim_code = 5
bookmark_code = 3
STIMULUS_EVENT_ID = 5
RESPONSE_EVENT_ID = 6
ERP_TMIN, ERP_TMAX = -0.2, 0.8
BASELINE = (-0.2, 0)
REJECT_CRITERIA = dict(eeg=100e-6)

# --- Plotting Configuration ---
# This list is used by create_erp_visualizations, which creates a combined Fz, Cz, Pz plot.
ERP_CHANNELS_TO_PLOT = ['Fz', 'Cz', 'Pz']
PLOT_INDIVIDUAL_LEGENDS = True

# --- Statistics Configuration ---
RUN_ERP_STATS = True
STAT_ALPHA = 0.05
N_PERMUTATIONS = 1024

# --- Peak Metrics Configuration ---
OZ_PEAK_CHANNEL = 'Oz'
OZ_PEAK_MODE = 'neg'
OZ_PEAK_TMIN = 0.150
OZ_PEAK_TMAX = 0.400

FZ_PEAK_CHANNEL = 'Fz'
FZ_PEAK_MODE = 'pos'
FZ_PEAK_TMIN = 0.150
FZ_PEAK_TMAX = 0.400

GFP_PEAK_TMIN = 0.150
GFP_PEAK_TMAX = 0.450


# === 2. HELPER FUNCTIONS ===
def extract_condition_name(wav_path):
    base = os.path.basename(str(wav_path))
    name = base.replace('.wav', '').replace('sounds_2/', '').lower()
    return 'silence' if name == 'silent2' else name


def get_condition_labels(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if 43 >= df.shape[1]:
        raise ValueError(f"Missing expected audio path column in {csv_path}")
    wav_paths = df[43].dropna().astype(str)
    wav_paths = wav_paths[wav_paths.str.endswith('.wav')]
    cleaned = wav_paths.apply(extract_condition_name).reset_index(drop=True)
    return cleaned.groupby((cleaned != cleaned.shift()).cumsum()).first().tolist()


def get_condition_segments(raw, condition_labels):
    events = mne.find_events(raw, stim_channel=stim_channel, initial_event=True, verbose=False)
    five_events = events[events[:, 2] == first_stim_code]
    three_events = events[events[:, 2] == bookmark_code]
    if not five_events.size or not three_events.size:
        return None, None, None
    trials_per_condition = len(five_events) // len(condition_labels)
    if trials_per_condition == 0:
        return None, None, None
    starts = [five_events[i * trials_per_condition, 0] for i in range(len(condition_labels))]
    ends = []
    for s in starts:
        following_threes = three_events[three_events[:, 0] > s]
        ends.append(following_threes[0, 0] - 1 if len(following_threes) > 0 else raw.n_times - 1)
    if len(starts) != len(condition_labels):
        return None, None, None
    return starts, ends, events


# =============================================================================
# === 3. MAIN DATA PROCESSING ===
# =============================================================================
def process_erp_data(common_channels):
    participant_evokeds = defaultdict(dict)
    for p_num in P_NUMBERS:
        print(f"\n{'=' * 20} Participant {p_num} {'=' * 20}")
        try:
            fif_path = os.path.join(base_data_path, f'Participant{p_num}_cleaned_raw.fif')
            csv_path = os.path.join(base_data_path, f'VisualResult_{p_num}.csv.csv')
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
            condition_labels = get_condition_labels(csv_path)

            starts, ends, all_events = get_condition_segments(raw, condition_labels)
            if starts is None:
                print(f"⚠️ Skipping P{p_num} (bad segmentation).")
                continue

            raw.pick(common_channels)
            raw.info['bads'] = []
            new_events_list, event_id_map, next_event_code = [], {}, 1

            # Process only the event-related conditions from the labels file
            for i, cond_name in enumerate(condition_labels):
                if i >= len(starts): break
                cond_start, cond_end = starts[i], ends[i]
                for event_type, event_id_val in [('stim', STIMULUS_EVENT_ID), ('resp', RESPONSE_EVENT_ID)]:
                    event_name = f"{cond_name}/{event_type}"
                    mask = (all_events[:, 0] >= cond_start) & (all_events[:, 0] <= cond_end) & (
                            all_events[:, 2] == event_id_val)
                    if np.any(mask):
                        if event_name not in event_id_map:
                            event_id_map[event_name] = next_event_code
                            next_event_code += 1
                        events_to_add = all_events[mask].copy()
                        events_to_add[:, 2] = event_id_map[event_name]
                        new_events_list.append(events_to_add)

            if not new_events_list:
                print(f"⚠️ Skipping P{p_num} (no valid events found).")
                continue

            new_events = np.vstack(new_events_list)
            new_events = new_events[new_events[:, 0].argsort()]

            epochs = mne.Epochs(raw, new_events, event_id=event_id_map, tmin=ERP_TMIN, tmax=ERP_TMAX,
                                baseline=BASELINE, reject=REJECT_CRITERIA, reject_by_annotation=True,
                                preload=True, verbose=False, event_repeated='drop')

            for event_name in epochs.event_id.keys():
                if len(epochs[event_name]) > 0:
                    participant_evokeds[p_num][event_name] = epochs[event_name].average()

        except Exception as e:
            print(f"❌ Failed for P{p_num}: {e}")

    return participant_evokeds


# === 4. PLOTTING FUNCTIONS ===
def create_erp_visualizations(grand_averages):
    print("\n" + "=" * 80 + "\n--- GENERATING GROUP ERP WAVEFORM PLOTS (Fz, Cz, Pz combined) ---\n" + "=" * 80)
    all_conditions_keys = set(cond for group in grand_averages.values() for cond in group)
    for cond_key in all_conditions_keys:
        plot_dict = {g: d[cond_key] for g, d in grand_averages.items() if cond_key in d}
        if not plot_dict: continue
        fig = mne.viz.plot_compare_evokeds(plot_dict, picks=ERP_CHANNELS_TO_PLOT, # Uses ERP_CHANNELS_TO_PLOT: ['Fz', 'Cz', 'Pz']
                                           title=cond_key.replace('/', ' ').title(), show=False, ci=0.95)
        fig[0].savefig(os.path.join(output_path, f'erp_waveform_{cond_key.replace("/", "_")}.png'), dpi=300)
        plt.close(fig[0])

def plot_channel_specific_comparison(grand_averages, channel_to_plot, output_path):
    print(f"\n--- Generating comparison plots for a single channel: '{channel_to_plot}' ---")
    ga_non_adhd = grand_averages.get('non-ADHD', {})
    ga_adhd = grand_averages.get('ADHD', {})

    for condition in CONDITIONS:
        stim_condition_key = f"{condition}/stim"
        evk_nonadhd = ga_non_adhd.get(stim_condition_key)
        evk_adhd = ga_adhd.get(stim_condition_key)

        if evk_adhd and evk_nonadhd:
            plot_dict = {'ADHD': evk_adhd, 'non-ADHD': evk_nonadhd}
            fig, ax = plt.subplots()
            mne.viz.plot_compare_evokeds(
                plot_dict, picks=channel_to_plot,
                title=f"ERP at {channel_to_plot}: {condition.replace('_', ' ').title()}",
                show=False, ci=0.95, colors={'ADHD': 'red', 'non-ADHD': 'blue'},
                linestyles={'ADHD': '-', 'non-ADHD': '--'}, axes=ax
            )
            ax.set_ylabel('Amplitude (µV)')
            ax.legend(title='Group')
            fig.savefig(os.path.join(output_path, f'erp_comparison_{channel_to_plot}_{condition}.png'), dpi=300)
            plt.close(fig)
        else:
            print(f"  Skipping plot for '{condition}': Missing data for one or both groups.")


def plot_gfp_with_individual_participants(participant_evokeds, grand_averages, output_path):
    print("\n--- Generating GFP plots with individual participant overlays (Manual Method) ---")
    adhd_p_nums = [p for p, g in GROUP_MAPPING.items() if g == 'ADHD' and p in participant_evokeds]
    non_adhd_p_nums = [p for p, g in GROUP_MAPPING.items() if g == 'non-ADHD' and p in participant_evokeds]

    cmap_adhd = plt.get_cmap('Reds')
    adhd_colors = {p_num: cmap_adhd(i) for i, p_num in enumerate(np.linspace(0.4, 0.9, len(adhd_p_nums)))}
    cmap_non_adhd = plt.get_cmap('Blues')
    non_adhd_colors = {p_num: cmap_non_adhd(i) for i, p_num in enumerate(np.linspace(0.4, 0.9, len(non_adhd_p_nums)))}
    participant_colors = {**adhd_colors, **non_adhd_colors}

    for condition in CONDITIONS:
        stim_condition_key = f"{condition}/stim"
        has_data = any(stim_condition_key in p_data for p_data in participant_evokeds.values())
        if not has_data: continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for p_num, p_data in participant_evokeds.items():
            if stim_condition_key in p_data:
                evoked = p_data[stim_condition_key]
                gfp = np.std(evoked.get_data(), axis=0)
                label = f'P{p_num}' if PLOT_INDIVIDUAL_LEGENDS else '_nolegend_'
                ax.plot(evoked.times, gfp * 1e6, color=participant_colors.get(p_num), alpha=0.7, label=label)

        group_primary_colors = {'ADHD': 'darkred', 'non-ADHD': 'darkblue'}
        for group, ga_dict in grand_averages.items():
            if stim_condition_key in ga_dict:
                ga_evoked = ga_dict[stim_condition_key]
                ga_gfp = np.std(ga_evoked.get_data(), axis=0)
                ax.plot(ga_evoked.times, ga_gfp * 1e6, color=group_primary_colors[group], label=f'{group} Grand Avg',
                        linewidth=3.0)

        ax.set_title(f"Global Field Power (GFP): {condition.replace('_', ' ').title()}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GFP (µV)')
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(True, linestyle='--', alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        ga_handles = [h for h, l in zip(handles, labels) if "Grand Avg" in l]
        ga_labels = [l for l in labels if "Grand Avg" in l]
        ax.legend(ga_handles, ga_labels, title='Group')
        fig.tight_layout()
        clean_fig_path = os.path.join(output_path, f'gfp_individual_overlay_{condition}.png')
        fig.savefig(clean_fig_path, dpi=300)

        if PLOT_INDIVIDUAL_LEGENDS:
            ax.legend(title='Participant / Group', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                      fontsize='small')
            plt.subplots_adjust(right=0.70)
            labeled_fig_path = os.path.join(output_path, f'gfp_individual_overlay_{condition}_labeled.png')
            fig.savefig(labeled_fig_path, dpi=300)
            print(f"  ... Saved diagnostic labeled plot for '{condition}'")

        plt.close(fig)
    print("  ...GFP plots saved.")

def plot_individual_channel_erp_comparisons(grand_averages, output_path):
    print("\n" + "=" * 80 + "\n--- GENERATING INDIVIDUAL CHANNEL ERP COMPARISON PLOTS (Fz, Cz, Pz, Oz) ---\n" + "=" * 80)
    # Define the list of channels to plot individually
    channels_to_plot_individually = ['Fz', 'Cz', 'Pz', 'Oz']
    colors = {'ADHD': 'red', 'non-ADHD': 'blue'}
    linestyles = {'ADHD': '-', 'non-ADHD': '--'}

    for condition in CONDITIONS:
        stim_condition_key = f"{condition}/stim"
        ga_adhd = grand_averages.get('ADHD', {}).get(stim_condition_key)
        ga_non_adhd = grand_averages.get('non-ADHD', {}).get(stim_condition_key)

        if not ga_adhd or not ga_non_adhd:
            print(f"  Skipping individual channel plots for '{condition}': Missing data for one or both groups.")
            continue

        plot_dict = {'ADHD': ga_adhd, 'non-ADHD': ga_non_adhd}

        for channel in channels_to_plot_individually:
            fig, ax = plt.subplots(figsize=(8, 6)) # Create a new figure for each channel
            mne.viz.plot_compare_evokeds(
                plot_dict, picks=channel, axes=ax,
                title=f"ERP at {channel}: {condition.replace('_', ' ').title()}",
                show=False, ci=0.95,
                colors=colors, linestyles=linestyles, legend=True # Ensure legend is shown for each plot
            )
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (µV)')
            ax.grid(True, linestyle=':')
            fig.tight_layout() # Ensure layout is tight for individual plots

            # Save the figure with a clear filename
            fig_path = os.path.join(output_path, f'erp_waveform_{channel}_{condition.replace("/", "_")}.png')
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            print(f"  ... Saved individual ERP plot for {channel} in {condition}.")

    print("\n--- All individual channel ERP plots saved. ---")


def plot_specific_2x2_erp_summary(grand_averages, output_path):
    print("\n" + "=" * 80 + "\n--- GENERATING SPECIFIC 2x2 ERP SUMMARY PLOT (Fz, Oz for High Intensity/Reverb & White Noise) ---\n" + "=" * 80)

    # Define the specific channels and conditions for this plot
    channels = ['Fz', 'Oz']
    conditions_to_plot = ['highintensity_highreverb', 'white_noise']
    colors = {'ADHD': 'red', 'non-ADHD': 'blue'}
    linestyles = {'ADHD': '-', 'non-ADHD': '--'}

    # Helper to format condition names for titles (reuse if available from other plotting funcs)
    def format_condition_for_plot(condition_name):
        if condition_name == 'white_noise':
            return 'White Noise'
        if condition_name == 'silence':
            return 'Silence'
        s = condition_name.replace('intensity', ' Intensity').replace('reverb', ' Reverb')
        s = s.replace('_', ', ')
        return s.title() + '.'

    fig, axes = plt.subplots(len(channels), len(conditions_to_plot), figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle('ERP Comparison: Fz & Oz for Specific Conditions', fontsize=16, weight='bold')

    for row_idx, channel in enumerate(channels):
        for col_idx, condition in enumerate(conditions_to_plot):
            ax = axes[row_idx, col_idx]
            stim_condition_key = f"{condition}/stim"

            ga_adhd = grand_averages.get('ADHD', {}).get(stim_condition_key)
            ga_non_adhd = grand_averages.get('non-ADHD', {}).get(stim_condition_key)

            if not ga_adhd or not ga_non_adhd:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"{channel} - {format_condition_for_plot(condition)}")
                ax.axis('off')
                continue

            plot_dict = {'ADHD': ga_adhd, 'non-ADHD': ga_non_adhd}

            mne.viz.plot_compare_evokeds(
                plot_dict, picks=channel, axes=ax,
                title=f"{channel} - {format_condition_for_plot(condition)}",
                show=False, ci=0.95,
                colors=colors, linestyles=linestyles, legend=False # Legend will be added once for the whole figure
            )
            ax.grid(True, linestyle=':')

            # Add axis labels to the bottom-most row and left-most column
            if row_idx == len(channels) - 1:
                ax.set_xlabel('Time (s)')
            if col_idx == 0:
                ax.set_ylabel('Amplitude (µV)')

    # Add a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Group', loc='upper right', bbox_to_anchor=(0.98, 0.95))

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.92]) # Adjust rect to make space for suptitle and legend
    fig_path = os.path.join(output_path, 'erp_2x2_Fz_Oz_HighReverb_WhiteNoise.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"\n--- Specific 2x2 ERP summary plot saved to: {fig_path} ---")


def plot_gfp_peak_topoplots_summary(grand_averages, output_path):
    """
    MODIFIED: Finds the peak GFP, plots scalp topography in a summary figure.
    - Condition names are formatted and slanted (e.g., 'Low intensity, low reverb.').
    - The main figure title is removed.
    """
    print("\n--- Generating HORIZONTAL Summary of Topoplots at GFP Peak (MODIFIED) ---")

    def format_condition_for_plot(condition_name):
        """Formats 'lowintensity_lowreverb' to 'Low intensity, low reverb.'."""
        if condition_name == 'white_noise':
            return 'White noise'
        if condition_name == 'silence':
            return 'Silence'

        # Add spaces before 'intensity' and 'reverb', then replace underscore
        s = condition_name.replace('intensity', ' intensity').replace('reverb', ' reverb')
        s = s.replace('_', ', ')

        # Capitalize first letter and add period
        return s.capitalize() + '.'

    # Calculate global min/max for consistent color scaling
    vlim_min, vlim_max = (np.inf, -np.inf)
    for group in GROUPS:
        for condition in CONDITIONS:
            stim_key = f"{condition}/stim"
            if stim_key in grand_averages.get(group, {}):
                evk = grand_averages[group][stim_key]
                vlim_min = min(vlim_min, evk.data.min())
                vlim_max = max(vlim_max, evk.data.max())
    vlim = max(abs(vlim_min), abs(vlim_max)) * 1e6

    n_groups = len(GROUPS)
    n_conditions = len(CONDITIONS)
    # Increased figure height slightly to accommodate rotated titles
    fig, axes = plt.subplots(n_groups, n_conditions, figsize=(3.5 * n_conditions, 4.5 * n_groups), squeeze=False)

    # MODIFICATION: Main title removed per user request
    # fig.suptitle('Scalp Topography at Peak Global Field Power (GFP)', fontsize=20, weight='bold')

    for row, group in enumerate(GROUPS):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            stim_key = f"{condition}/stim"

            # Set group name as Y-axis label for the first column
            if col == 0:
                ax.set_ylabel(group, fontsize=16, fontweight='bold', labelpad=40)

            evk = grand_averages.get(group, {}).get(stim_key)
            if evk:
                try:
                    # Find GFP peak to plot the topography
                    _, peak_time, _ = evk.get_peak(ch_type='eeg', tmin=GFP_PEAK_TMIN, tmax=GFP_PEAK_TMAX, mode='pos',
                                                   return_amplitude=True)

                    # Plot the topomap
                    evk.plot_topomap(times=peak_time, axes=ax, show=False, vlim=(-vlim, vlim), colorbar=False)

                    # MODIFICATION: Set slanted, formatted titles only for the top row
                    if row == 0:
                        condition_title = format_condition_for_plot(condition)
                        ax.set_title(condition_title, fontsize=14, rotation=30, ha='center', y=1.15)
                    else:
                        ax.set_title("")  # No title for subsequent rows

                    # Use xlabel for peak time on all plots for consistency
                    ax.set_xlabel(f"Peak at {peak_time * 1000:.0f} ms", fontsize=10)

                except Exception as e:
                    ax.text(0.5, 0.5, 'Error plotting', ha='center', va='center')
                    ax.axis('off')
                    print(f"Could not plot topomap for {group}/{condition}: {e}")
            else:
                # Handle cases with no data
                if row == 0:
                    condition_title = format_condition_for_plot(condition)
                    ax.set_title(condition_title, fontsize=14, rotation=30, ha='center', y=1.15)
                else:
                    ax.set_title("")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.axis('off')

    # Add a single, shared colorbar for the entire figure
    mappable = None
    for ax in axes.ravel():
        if ax.images:
            mappable = ax.images[0]
            break

    if mappable:
        # Adjust layout to make space for colorbar and rotated titles
        fig.tight_layout(rect=[0.05, 0.05, 0.92, 0.92])
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Amplitude (µV)', fontsize=12)
    else:
        print("Warning: No data was plotted, so no colorbar will be generated.")
        fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig_path = os.path.join(output_path, 'summary_topoplots_at_gfp_peak_HORIZONTAL.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  ...Saved HORIZONTAL GFP peak topoplot summary to:\n  {fig_path}")


# === 5. STATISTICS (CLUSTER-BASED) ===
# MODIFIED with a manual threshold for more sensitive analysis
def run_erp_statistics(participant_evokeds, info_template):
    if not RUN_ERP_STATS: return
    print("\n" + "=" * 80 + "\n--- RUNNING CLUSTER STATS (WITH CACHING) ---\n" + "=" * 80)

    try:
        adjacency, _ = mne.channels.find_ch_adjacency(info_template, ch_type='eeg')
    except Exception as e:
        print(f"Adjacency setup failed: {e}. Cannot run cluster stats.");
        return

    # We use the t-value corresponding to p=0.05 for a two-tailed test.
    from scipy import stats
    # Degrees of freedom will be (n_adhd - 1) + (n_non_adhd - 1)
    n_adhd = sum(1 for g in GROUP_MAPPING.values() if g == 'ADHD')
    n_non_adhd = sum(1 for g in GROUP_MAPPING.values() if g == 'non-ADHD')
    degrees_of_freedom = n_adhd + n_non_adhd - 2
    # For a two-tailed p=0.05, we look at the 97.5th percentile (0.025 in each tail)
    t_threshold = stats.t.ppf(1 - 0.05 / 2, df=degrees_of_freedom)
    print(f"--- Using a manual t-threshold of: {t_threshold:.3f} (for p=0.05) ---")
    # --------------------------------------------------------

    for condition in CONDITIONS:
        print(f"\n--- Running stats for condition: {condition} ---")
        g1_evokeds = [p_data.get(f"{condition}/stim") for p, p_data in participant_evokeds.items() if
                      GROUP_MAPPING[p] == 'ADHD']
        g2_evokeds = [p_data.get(f"{condition}/stim") for p, p_data in participant_evokeds.items() if
                      GROUP_MAPPING[p] == 'non-ADHD']
        g1 = [evk for evk in g1_evokeds if evk is not None]
        g2 = [evk for evk in g2_evokeds if evk is not None]

        if len(g1) < 2 or len(g2) < 2: continue

        cache_path = os.path.join(STATS_CACHE_DIR, f'stats_cache_{condition.replace("/", "_")}.pkl')
        if os.path.exists(cache_path) and USE_PROCESSED_DATA_CACHE:
            with open(cache_path, 'rb') as f:
                stats_data = pickle.load(f)
            t_obs, clusters, p_vals = stats_data['t_obs'], stats_data['clusters'], stats_data['p_vals']
        else:
            X1 = np.array([e.get_data().transpose(1, 0) for e in g1])
            X2 = np.array([e.get_data().transpose(1, 0) for e in g2])

            # --- MODIFIED: Pass the manual threshold to the function ---
            t_obs, clusters, p_vals, _ = mne.stats.permutation_cluster_test(
                [X1, X2],
                stat_fun=ttest_ind_no_p,
                n_permutations=N_PERMUTATIONS,
                adjacency=adjacency,
                n_jobs=-1,
                tail=0,
                out_type='mask',
                threshold=t_threshold  # This is the crucial addition
            )
            with open(cache_path, 'wb') as f:
                pickle.dump({'t_obs': t_obs, 'clusters': clusters, 'p_vals': p_vals}, f)

        good_cluster_idxs = np.where(p_vals < STAT_ALPHA)[0]
        if not len(good_cluster_idxs):
            print(f"  No significant clusters found for '{condition}'.")
            continue

        print(f"  ✅ Found {len(good_cluster_idxs)} significant clusters in '{condition}'")
        for i, idx in enumerate(good_cluster_idxs):
            cluster_mask = clusters[idx]
            diff = mne.combine_evoked([mne.grand_average(g1), mne.grand_average(g2)], weights=[1, -1])
            fig = diff.plot_image(mask=cluster_mask.T, show=False, titles=f"p-value: {p_vals[idx]:.4f}", mask_alpha=0.5)
            fig.suptitle(f'Group Difference: {condition} - Cluster {i + 1}')
            fig.savefig(os.path.join(output_path, f'stats_group_{condition.replace("/", "_")}_cluster_{i + 1}.png'),
                        dpi=300)
            plt.close(fig)


# === 6. STATISTICS (ERP METRICS) ===
def calculate_peak_metrics(participant_evokeds):
    print("\n" + "=" * 80 + "\n--- GENERATING PEAK METRICS DATA FOR R ANALYSIS ---\n" + "=" * 80)
    analyses_to_run = [
        {'name': 'oz', 'channel': OZ_PEAK_CHANNEL, 'mode': OZ_PEAK_MODE, 'tmin': OZ_PEAK_TMIN, 'tmax': OZ_PEAK_TMAX},
        {'name': 'fz', 'channel': FZ_PEAK_CHANNEL, 'mode': FZ_PEAK_MODE, 'tmin': FZ_PEAK_TMIN, 'tmax': FZ_PEAK_TMAX},
        # NEW: Negative potential of Fz from 200 to 400ms
        {'name': 'fz_neg_200_400', 'channel': 'Fz', 'mode': 'neg', 'tmin': 0.200, 'tmax': 0.400},
        # NEW: Positive potential of Oz from 100 to 200ms
        {'name': 'oz_pos_100_200', 'channel': 'Oz', 'mode': 'pos', 'tmin': 0.100, 'tmax': 0.300},
    ]
    results = []
    for p_num, p_data in participant_evokeds.items():
        group = GROUP_MAPPING[p_num]
        for cond_name in CONDITIONS:
            event_name = f"{cond_name}/stim"
            if event_name in p_data:
                evoked = p_data[event_name]
                result_row = {'participant': p_num, 'group': group, 'condition': cond_name}
                for analysis in analyses_to_run:
                    an_name = analysis['name']
                    try:
                        _, lat_s, amp_v = evoked.copy().pick(analysis['channel']).get_peak(tmin=analysis['tmin'],
                                                                                           tmax=analysis['tmax'],
                                                                                           mode=analysis['mode'],
                                                                                           return_amplitude=True)
                        result_row[f'{an_name}_peak_latency_ms'] = lat_s * 1000
                        result_row[f'{an_name}_peak_amplitude_uv'] = amp_v * 1e6
                    except Exception:
                        result_row[f'{an_name}_peak_latency_ms'], result_row[
                            f'{an_name}_peak_amplitude_uv'] = None, None
                try:
                    _, gfp_lat_s, gfp_amp_v = evoked.get_peak(ch_type='eeg', tmin=GFP_PEAK_TMIN, tmax=GFP_PEAK_TMAX,
                                                              mode='pos', return_amplitude=True)
                    result_row['gfp_peak_latency_ms'] = gfp_lat_s * 1000
                    result_row['gfp_peak_amplitude_uv'] = gfp_amp_v * 1e6
                except Exception:
                    result_row['gfp_peak_latency_ms'], result_row['gfp_peak_amplitude_uv'] = None, None
                results.append(result_row)
    return pd.DataFrame(results)


# === 7. SCRIPT EXECUTION ===
if __name__ == '__main__':
    if os.path.exists(PROCESSED_DATA_CACHE) and USE_PROCESSED_DATA_CACHE:
        print(f"--- Loading cached processed data from {PROCESSED_DATA_CACHE} ---")
        with open(PROCESSED_DATA_CACHE, 'rb') as f:
            participant_evokeds = pickle.load(f)
    else:
        print("--- No cache found or cache disabled. Starting data processing from scratch... ---")
        all_ch_names = []
        for p in P_NUMBERS:
            try:
                fif_path = os.path.join(base_data_path, f'Participant{p}_cleaned_raw.fif')
                all_ch_names.append(mne.io.read_raw_fif(fif_path, preload=False, verbose=False).ch_names)
            except FileNotFoundError:
                print(f"Warning: Could not find data for Participant {p}. Skipping for channel intersection.")
        if not all_ch_names: raise FileNotFoundError("Could not load any participant data.")
        common_channels = list(set.intersection(*map(set, all_ch_names)))
        common_channels = [ch for ch in common_channels if ch != stim_channel]
        print(f"Found {len(common_channels)} common channels across all participants.")
        participant_evokeds = process_erp_data(common_channels)
        with open(PROCESSED_DATA_CACHE, 'wb') as f:
            pickle.dump(participant_evokeds, f)
        print(f"--- Saved processed data to cache: {PROCESSED_DATA_CACHE} ---")

    info_template = None
    for p_data in participant_evokeds.values():
        if p_data: info_template = next(iter(p_data.values())).info; break
    if not info_template: raise RuntimeError("Could not find any evoked data to create info template.")

    grand_averages = defaultdict(dict)
    for group in GROUPS:
        p_nums_in_group = [p for p, g in GROUP_MAPPING.items() if g == group]
        all_evokeds_in_group = [participant_evokeds[p] for p in p_nums_in_group if p in participant_evokeds]
        all_cond_keys = set(key for p_data in all_evokeds_in_group for key in p_data)
        for cond_key in all_cond_keys:
            evokeds_to_average = [p_data[cond_key] for p_data in all_evokeds_in_group if cond_key in p_data]
            if evokeds_to_average:
                grand_averages[group][cond_key] = mne.grand_average(evokeds_to_average)

    # Call the original function that plots Fz, Cz, Pz combined (if desired)
    create_erp_visualizations(grand_averages)

    # These are already individual for Oz and Fz. Keep them if you want distinct plots for only these two.
    plot_channel_specific_comparison(grand_averages, OZ_PEAK_CHANNEL, output_path)
    plot_channel_specific_comparison(grand_averages, FZ_PEAK_CHANNEL, output_path)

    plot_gfp_with_individual_participants(participant_evokeds, grand_averages, output_path)

    # Call the NEW function to plot Fz, Cz, Pz, Oz as individual files (replaces old holistic summary)
    plot_individual_channel_erp_comparisons(grand_averages, output_path)

    # Call the NEW function for the specific 2x2 plot of Fz/Oz for selected conditions
    plot_specific_2x2_erp_summary(grand_averages, output_path)

    plot_gfp_peak_topoplots_summary(grand_averages, output_path)

    run_erp_statistics(participant_evokeds, info_template)

    metrics_df = calculate_peak_metrics(participant_evokeds)
    if not metrics_df.empty:
        output_csv_path = os.path.join(output_path, 'peak_metrics_results.csv')
        metrics_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Successfully saved combined metrics data to:\n{output_csv_path}")
    else:
        print("\n❌ Metrics DataFrame is empty. No peaks were found.")

    print("\n✅ All Python processing finished successfully.")
