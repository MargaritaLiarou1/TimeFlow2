import numpy as np
import pandas as pd
from collections import defaultdict
from tslearn.utils import to_time_series_dataset
import re

np.random.seed(1)


class PathTimeSeriesConstructor:

    def __init__(self, df):

        # represent paths as time series based on their clusters' (cell states) markers, pseudotime column is excluded
        self.df = df.select_dtypes(include=np.number).drop(columns=["pseudotime"])
        self.feature_count = self.df.shape[1]

    def path_group_time_series(self, groups_dict, indices_dict):

        # create a time series for each group of pathways
        # each time step corresponds to a segment and is characterized by a feature vector with dimensionality equal to the number of markers
        # marker values are aggregated over all cells in the cluster at a specific segment

        group_time_series = {}

        for group_name, cluster_entries in groups_dict.items():
            segment_cluster_map = defaultdict(list)
            # for each cluster extract its cluster and segment id
            for entry in cluster_entries:
                match = re.match(r"cluster_(\d+)_segment_S(\d+)", entry)
                if match:
                    cluster_id = int(match.group(1))
                    segment = int(match.group(2))
                    segment_cluster_map[segment].append(cluster_id)

            # store time series for the group in loop
            ts = []

            # loop over all segments and get the cluster ids
            for segment in sorted(segment_cluster_map.keys()):
                segment_key = f"S{segment}"
                clusters = segment_cluster_map[segment]

                # store in a list all cell indices related to the above clusters
                all_indices = []
                for cluster_id in clusters:
                    # cell indices for cluster and segment based on their keys
                    indices = indices_dict.get(segment_key, {}).get(cluster_id, [])
                    all_indices.extend(indices)

                # compute mean marker values based on all these cell indices
                if all_indices:
                    mean_vector = self.df.iloc[all_indices].mean().values
                else:
                    # fill with NaN if indices do not exist
                    mean_vector = np.full(self.feature_count, np.nan)
                ts.append(mean_vector)

            # store mean vectors for segments of group in loop
            group_time_series[group_name] = ts

        return group_time_series

    def ts_learn_convertion(self, grouped_data):

        # https://tslearn.readthedocs.io/en/stable/gen_modules/utils/tslearn.utils.to_time_series_dataset.html#tslearn.utils.to_time_series_dataset
        # convert paths to a ts-learn dataset (shape: number of paths, max time steps, number of markers)
        time_series_list = [np.vstack(group) for group in grouped_data.values()]

        return to_time_series_dataset(time_series_list)

    def cluster_segment_reformatting(self, value):

        # path strings formatted as cluster_c_segment_s to print the groups
        segments = re.findall(r'segment (\d+), cluster (\d+)', value)
        return ' -> '.join([f'cluster_{cluster}_segment_S{segment}' for segment, cluster in segments])

    def group_unique_segment_extraction(self, grouped_dict, paths_dict):

        # extract all cluster_c_segment_s for paths in each group

        result = {}

        for group_name, keys in grouped_dict.items():
            # extract backtracked paths only for clusters/segments belonging to the group in loop
            valid_paths = {key: paths_dict[key] for key in keys if key in paths_dict}
            reformatted = {key: self.cluster_segment_reformatting(val) for key, val in valid_paths.items()}

            unique_segments = set()
            # loop over the above backtracked paths
            for path in reformatted.values():
                # split by arrow
                segments = path.split(" -> ")
                # get unique segments
                unique_segments.update(segments)

            result[group_name] = sorted(unique_segments)

        return result

    def empty_groups_removal(self, grouped_dict):

        # redundant when referring at path level, there are no duplicate states in this case
        return {group: clusters for group, clusters in grouped_dict.items() if clusters}
