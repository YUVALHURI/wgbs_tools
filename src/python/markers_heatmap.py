import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from beta_to_table import beta2table_generator
from itertools import accumulate
from utils_wgbs import COORDS_COLS5
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import scipy.cluster.hierarchy as sch

MARKERS_FILES_COLUMNS = COORDS_COLS5 + ['target', 'region', 'lenCpG', 'bp', 'tg_mean', 'bg_mean', 'delta_means', 'delta_quants', 'delta_maxmin', 'ttest', 'direction']


def read_groups(group_file, groups_filter):
    groups_df = pd.read_csv(group_file)

    if not groups_filter:
        groups_filter = list(dict.fromkeys(groups_df['group'].tolist()))

    # Groups are sorted by the csv order or the filter, then samples blocks and group markers correspond to that order
    groups_to_plot_df = groups_df[groups_df['group'].isin(groups_filter)]
    groups_to_plot_df['group'] = pd.Categorical(groups_to_plot_df['group'], categories=groups_filter, ordered=True)
    groups_to_plot_df = groups_to_plot_df.sort_values('group')
    return groups_to_plot_df


def create_marker_regions_df_from_custom_file(custom_marker_file):
    # todo: what to do with these extra 2 columns?
    marker_regions_df = pd.read_csv(custom_marker_file, sep='\t', header=None, names=MARKERS_FILES_COLUMNS + ['anno',
                                                                                                              'gene'])
    marker_regions_df['group'] = 'ungrouped'
    return marker_regions_df


def create_marker_regions_df_from_markers_dir(markers_files_dir, groups_list, group_blocks_indices, markers_per_group=25):
    marker_regions_df = pd.DataFrame(columns=['group', 'chr', 'start', 'end', 'startCpG', 'endCpG'])
    for group in groups_list:
        markers_file = os.path.join(markers_files_dir, f'Markers.{group}.bed')
        markers_df = pd.read_csv(markers_file, sep='\t', header=None, comment='#', names=MARKERS_FILES_COLUMNS)
        markers_df['group'] = group
        top_markers_df = markers_df.head(markers_per_group)
        marker_regions_df = pd.concat(
            [marker_regions_df, top_markers_df[['group', 'chr', 'start', 'end', 'startCpG', 'endCpG']]],
            ignore_index=True)
        group_blocks_indices.append(group_blocks_indices[-1] + len(top_markers_df))
    return marker_regions_df


def create_markers_heat_map_uxm(marker_regions_df, samples_list, uxm_files_dir, marker_type='U'):

    marker_column_names = marker_regions_df.apply(lambda row: f"{row['group']}.marker.{row['chr']}:{row['start']}-{row['end']}", axis=1).tolist()
    markers_to_samples_methylation_df = pd.DataFrame(columns=marker_column_names)
    for sample in samples_list:
        uxm_file = os.path.join(uxm_files_dir, f'{sample}.uxm.bed.gz')
        print(f'Collecting data for file {uxm_file}...')
        uxm_df = pd.read_csv(uxm_file, sep='\t', header=None, names=['chr', 'start', 'end', 'startCpG', 'endCpG', 'U', 'X', 'M'])
        uxm_df_markers = uxm_df.merge(marker_regions_df, on=['chr','start','end','startCpG', 'endCpG'], how='inner')
        uxm_df_markers['value'] = uxm_df_markers[marker_type]/(uxm_df_markers['X']+uxm_df_markers['M']+uxm_df_markers['U'])
        file_marker_column_names = uxm_df_markers.apply(lambda row: f"{row['group']}.marker.{row['chr']}:{row['start']}-{row['end']}", axis=1).tolist()
        uxm_df_markers_transposed = uxm_df_markers[['value']].transpose()
        uxm_df_markers_transposed.index = [sample]
        uxm_df_markers_transposed.columns = file_marker_column_names
        assert set(marker_column_names) == set(file_marker_column_names) #todo: just for now

        markers_to_samples_methylation_df = pd.concat([markers_to_samples_methylation_df, uxm_df_markers_transposed])

    return markers_to_samples_methylation_df

    #######
    #todo: what about NaNs? are they shown as NaN or 0?
    ########


def create_markers_heat_map_betas_alternative(marker_regions_df, groups_list, samples_list, blocks_file, beta_files_dir):

    beta_files = [os.path.join(beta_files_dir, f'{sample}.beta') for sample in samples_list]
    # The indices where one group ends and another one starts, for visualization
    group_blocks_indices = [0]

    # Start by creating the transposed heatmap, in the end we will transpose it back.
    markers_to_samples_methylation_df = pd.DataFrame(columns=samples_list+['group'])
    markers_to_samples_methylation_df_index = []
    beta_to_table_df = beta2table_generator(beta_files, blocks_file, None, 4, 8, 200000, False)
    for chunk_df in beta_to_table_df:
        markers_in_chunk = chunk_df.merge(marker_regions_df, on=['chr', 'start', 'end', 'startCpG', 'endCpG'], how='inner')

        markers_to_samples_methylation_df = pd.concat(
            [markers_to_samples_methylation_df, markers_in_chunk[samples_list+['group']]], ignore_index=True)
        markers_to_samples_methylation_df_index += markers_in_chunk.apply(lambda row: f"{row['group']}.marker.{row['region']}", axis=1).tolist()

    markers_to_samples_methylation_df.index = markers_to_samples_methylation_df_index
    markers_to_samples_methylation_df['group'] = pd.Categorical(markers_to_samples_methylation_df['group'], categories=groups_list, ordered=True)
    markers_to_samples_methylation_df = markers_to_samples_methylation_df.sort_values('group')
    markers_to_samples_methylation_df = markers_to_samples_methylation_df.drop(columns=['group'])
    df_to_plot = markers_to_samples_methylation_df.transpose()
    return df_to_plot


def plot_heatmap(df, output_file_name, groups_list, group_blocks_indices, line_indices, auto_cluster_columns=False, reverse_colors=False):
    # Prepare plot
    plt.figure(figsize=(50, 50), dpi=150)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    colors = [(1, 1, 0), (1, 1, 1), (0, 0, 1)]
    if reverse_colors:
        colors = [(0, 0, 1),  (1, 1, 1), (1, 1, 0)]
    cmap = LinearSegmentedColormap.from_list('Custom', colors, N=256)
    cmap.set_bad(color='grey')

    ax = plt.gca()
    for line_index in line_indices:
        ax.axhline(y=line_index, color='black', linestyle='-', linewidth=1)
    for marker_index in group_blocks_indices:
        ax.axvline(x=marker_index, color='black', linestyle='-', linewidth=1)

    midpoints = [(0 + line_indices[0]) / 2] + [(line_indices[i] + line_indices[i + 1]) / 2 for i in
                                               range(len(line_indices) - 1)]

    for index, group in enumerate(groups_list):
        ax.text(len(df.columns)+1, midpoints[index], group, ha='left', va='center', fontsize=5,
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    if auto_cluster_columns:
        data_filled = df.fillna(0)

        distance_matrix = np.corrcoef(data_filled.T)  # Transpose to cluster columns
        distance_matrix = 1 - distance_matrix  # Convert correlation to distance

        linkage = sch.linkage(distance_matrix, method='ward', metric='euclidean')

        dendrogram = sch.dendrogram(linkage, no_plot=True)
        ordered_indices = dendrogram['leaves']
        ordered_column_names = [data_filled.columns[i] for i in ordered_indices]
        prefix_order = {prefix: index for index, prefix in enumerate(groups_list)}
        ordered_column_names = sorted(ordered_column_names, key=lambda item: prefix_order.get(item.split('.marker')[0], float('inf')))

        df = df[ordered_column_names]
    sns.heatmap(df, cmap=cmap, cbar=False, yticklabels=True, xticklabels=True)
    plt.tight_layout()
    plt.savefig(output_file_name, format="pdf", bbox_inches="tight")


def main():
    args = parse_args()
    groups_df = read_groups(args.groups_file, args.groups_filter)
    samples_list = groups_df['name'].tolist()
    groups_list = groups_df['group'].drop_duplicates().tolist()
    group_sizes = groups_df.groupby('group').size().tolist()
    line_indices = list(accumulate(group_sizes))
    group_blocks_indices = [0]

    if args.custom_markers_file is None:
        marker_regions_df = create_marker_regions_df_from_markers_dir(args.markers_dir, groups_list, group_blocks_indices, args.markers_per_group)
        # By default, columns will be created by the order of the corresponding marker file of the group, so that each group will have it's markers clustered together.
        auto_cluster_columns = False
    else:
        marker_regions_df = create_marker_regions_df_from_custom_file(args.custom_markers_file)
        auto_cluster_columns = True


    if args.uxms_dir:
        # todo: maybe create homogs from pats automatically?
        # Backup for me - saving df to csv to test only heatmap creation
        # df.to_csv("backup_heatmap.csv")
        # markers_to_samples_methylation_df = pd.read_csv('backup_heatmap.csv', index_col=0)
        markers_to_samples_methylation_df = create_markers_heat_map_uxm(marker_regions_df, samples_list, args.uxms_dir, args.direction)
        plot_heatmap(markers_to_samples_methylation_df, args.output, groups_list, group_blocks_indices, line_indices, auto_cluster_columns=auto_cluster_columns)

    elif args.betas_dir:
        markers_to_samples_methylation_df = create_markers_heat_map_betas_alternative(marker_regions_df, groups_list, samples_list, args.blocks_file, args.betas_dir)
        plot_heatmap(markers_to_samples_methylation_df, args.output, groups_list, group_blocks_indices, line_indices, auto_cluster_columns=auto_cluster_columns, reverse_colors=True)
    else:
        print('no uxms or betas dir given')


def parse_args():
    #todo: validate args
    #todo: allow custom groups clustering? (draw heatmap without defining the group lines)
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--groups_file', '-g', required=True,
                        help='groups csv file that was used in the marker files creation. this will not be used to average methylation over beta files, but rather to separate the markers by the groups' )
    parser.add_argument(
        '--markers_dir', '-md', help='Markers directory with markers files (Markers.*.Bed)')
    parser.add_argument('--betas_dir', '-bd',  help='beta files directory')
    parser.add_argument('--uxms_dir', '-ud', help='UXMS directory with UXMS files')
    parser.add_argument('--blocks_file', '-bf', help='blocks file')
    parser.add_argument(
        '--output', '-o', required=True, help='specify output path for the pdf')
    parser.add_argument(
        '--markers_per_group', '-mpg', type=int, default=25, help='groups to show in the final plot')
    parser.add_argument(
        '--groups_filter', '-gf',  type=list_of_strings, help='groups to show in the final plot')
    parser.add_argument(
        '--direction', '-d', help='direction - U/X/M', default='U')
    parser.add_argument(
        '--custom_markers_file', '-cmf', help='Custom markers file in case you dont have a clustered markers file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()