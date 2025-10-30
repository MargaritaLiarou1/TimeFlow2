import numpy as np
import pandas as pd
from collections import defaultdict
from tslearn.utils import to_time_series_dataset
import re

np.random.seed(1)


class InitialBackboneTimeSeries:

    def __init__(self, initial_pathway_groups, indices_dict, df, backtracked_paths):

        self.initial_pathway_groups = initial_pathway_groups
        self.indices_dict = indices_dict
        self.df = df
        self.backtracked_paths = backtracked_paths
        self.group_time_series = None
        self.ts_dataset = None

    def path_group_time_series(self):

        # create a time series for each group of pathways
        # each time step corresponds to a segment and is characterized by a feature vector with dimensionality equal to the number of markers
        # feature values are aggregated over all cells forming the clusters of the pathway at a specific segment

        df_numeric = self.df.select_dtypes(include=[np.number]).drop(columns=["pseudotime"])
        # count numeric features
        feature_count = df_numeric.shape[1]
        self.group_time_series = {}

        # loop over all groups and their clusters
        for group_name, cluster_entries in self.initial_pathway_groups.items():

            # group clusters by their segment
            segment_cluster_map = defaultdict(list)
            # for each cluster extract its cluster and segment id
            for entry in cluster_entries:
                match = re.match(r"cluster_(\d+)_segment_S(\d+)", entry)
                if match:
                    # if match true extract cluster and segment id
                    cluster_id = int(match.group(1))
                    segment = int(match.group(2))
                    # append the above ids in a list for segment in loop
                    segment_cluster_map[segment].append(cluster_id)

            # empty list to store time series for group in loop
            ts = []
            # loop over all segments and get relevant cluster ids
            for segment in sorted(segment_cluster_map.keys()):
                segment_key = f"S{segment}"
                clusters = segment_cluster_map[segment]

                # store in a list all cell indices relevant to the above clusters
                all_indices = []
                for cluster_id in clusters:
                    # cell indices for cluster and segment based on their keys
                    indices = self.indices_dict.get(segment_key, {}).get(cluster_id, [])
                    all_indices.extend(indices)

                # if indices exist compute mean feature values based on all these cell indices
                if all_indices:
                    mean_vector = df_numeric.iloc[all_indices].mean().values
                    # if indices do not exist fill with NaN
                else:
                    mean_vector = np.full(feature_count, np.nan)
                ts.append(mean_vector)

            # store mean vectors for segments of group in loop
            self.group_time_series[group_name] = ts

        return self.group_time_series

    def ts_learn_convertion(self):

        # https://tslearn.readthedocs.io/en/stable/gen_modules/utils/tslearn.utils.to_time_series_dataset.html#tslearn.utils.to_time_series_dataset
        # treat each group as a time series
        time_series_list = [np.vstack(group) for group in self.group_time_series.values()]

        # convert paths to a ts-learn dataset (shape: number of paths, max time steps, number of markers)
        self.ts_dataset = to_time_series_dataset(time_series_list)

        return self.ts_dataset


