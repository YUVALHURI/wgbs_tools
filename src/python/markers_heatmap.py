import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from beta_to_table import beta2table_generator
from itertools import accumulate
from utils_wgbs import COORDS_COLS5
from matplotlib.colors import LinearSegmentedColormap

MARKERS_FILES_COLUMNS = COORDS_COLS5 + ['target', 'region', 'lenCpG', 'bp', 'tg_mean', 'bg_mean', 'delta_means', 'delta_quants', 'delta_maxmin', 'ttest', 'direction']


def create_markers_heat_map(markers_files_dir, blocks_file, beta_files_dir, groups_file, output_file_name, markers_per_group=25, groups_filter=None):
    groups_df = pd.read_csv(groups_file)

    if not groups_filter:
        groups_filter = list(dict.fromkeys(groups_df['group'].tolist()))

    # Groups are sorted by the csv order or the filter, then samples blocks and group markers correspond to that order
    groups_to_plot_df = groups_df[groups_df['group'].isin(groups_filter)]
    groups_to_plot_df['group'] = pd.Categorical(groups_to_plot_df['group'], categories=groups_filter, ordered=True)
    groups_to_plot_df = groups_to_plot_df.sort_values('group')
    samples_list = groups_to_plot_df['name'].tolist()
    groups_list = groups_to_plot_df['group'].drop_duplicates().tolist()

    markers_to_samples_methylation_df = pd.DataFrame(columns=samples_list)
    markers_to_samples_methylation_df_index = []

    markers_files = [os.path.join(markers_files_dir, f'Markers.{group}.bed') for group in groups_list]
    beta_files = [os.path.join(beta_files_dir, f'{sample}.beta') for sample in samples_list]

    group_blocks_indices = [0]  # The indices where one group ends and another one starts, for visualization

    for markers_file in markers_files:
        print(f'Collecting data for marker {markers_file}...')
        markers_df = pd.read_csv(markers_file, sep='\t', header=None, comment='#', names=MARKERS_FILES_COLUMNS)
        top_markers_df = markers_df.head(markers_per_group)
        beta_to_table_df = beta2table_generator(beta_files, blocks_file, None, 4, 8, 200000, False)

        # Collect samples data for the top markers regions, in chunks
        group_markers_regions = []
        for chunk_df in beta_to_table_df:
            markers_in_chunk = chunk_df.merge(top_markers_df, on=['startCpG', 'endCpG'], how='inner')

            # Concat the results to the big heatmap. It's important to note that the different markers will not be added by their order, but by region.
            markers_to_samples_methylation_df = pd.concat([markers_to_samples_methylation_df, markers_in_chunk[samples_list]], ignore_index=True)
            group_markers_regions += markers_in_chunk['region'].tolist()

        # Add new markers to the heatmap
        markers_to_samples_methylation_df_index += [f"{os.path.basename(markers_file).split('.')[1]}.marker.{r}" for r in group_markers_regions]
        group_blocks_indices.append(group_blocks_indices[-1] + len(group_markers_regions))

        print(f'Finished collecting data for marker {markers_file}')

    print('Finished collecting data for all markers')

    markers_to_samples_methylation_df.index = markers_to_samples_methylation_df_index

    # Prepare plot
    plt.figure(figsize=(50, 50), dpi=150)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    colors = [(0, 0, 1), (1, 1, 1), (1, 1, 0)]
    cmap = LinearSegmentedColormap.from_list('Custom', colors, N=256)
    cmap.set_bad(color='grey')

    group_sizes = groups_to_plot_df.groupby('group').size().tolist()
    line_indices = list(accumulate(group_sizes))
    ax = plt.gca()
    for line_index in line_indices:
        ax.axhline(y=line_index, color='black', linestyle='-', linewidth=1)
    for marker_index in group_blocks_indices:
        ax.axvline(x=marker_index, color='black', linestyle='-', linewidth=1)

    midpoints = [(0 + line_indices[0]) / 2] + [(line_indices[i] + line_indices[i + 1]) / 2 for i in range(len(line_indices) - 1)]

    for index, group in enumerate(groups_list):
        ax.text(group_blocks_indices[-1]+1, midpoints[index], group, ha='left', va='center', fontsize=5,
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    df_transposed = markers_to_samples_methylation_df.transpose()

    # Backup for me - saving df to csv to test only visuals
    df_transposed.to_csv("backup_heatmap.csv")

    sns.heatmap(df_transposed, cmap=cmap, cbar=False, yticklabels=True, xticklabels=True)
    plt.tight_layout()
    plt.savefig(output_file_name, format="pdf", bbox_inches="tight")


def main():
    args = parse_args()
    create_markers_heat_map(args.markers_dir, args.blocks_file, args.betas_dir, args.groups_file, args.output, args.markers_per_group, args.groups_filter)



def parse_args():
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--groups_file', '-g', required=True,
                        help='groups csv file that was used in the marker files creation. this will not be used to average methylation over beta files, but rather to separate the markers by the groups' )
    parser.add_argument(
        '--markers_dir', '-md', required=True, help='Markers directory with markers files (Markers.*.Bed)')
    parser.add_argument('--betas_dir', '-bd',  required=True,  help='beta files directory')
    parser.add_argument('--blocks_file', '-bf',  required=True,  help='blocks file')
    parser.add_argument(
        '--output', '-o', required=True, help='specify output path for the pdf')
    parser.add_argument(
        '--markers_per_group', '-mpg', type=int, default=25, help='groups to show in the final plot')
    parser.add_argument(
        '--groups_filter', '-gf',  type=list_of_strings, help='groups to show in the final plot')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()