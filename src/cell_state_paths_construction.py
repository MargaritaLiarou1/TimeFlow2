import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import itertools
import re

np.random.seed(1)


class CellStatePathConstructor:

    def __init__(self, df, pseudotime_col='pseudotime', num_segments=10, overlap_fraction=0.35, max_clusters=25):

        # apart from cytometry markers the dataset needs a pseudotime column, estimated by TimeFlow 1 (https://github.com/MargaritaLiarou1/TimeFlow)
        self.df_original = df.copy()
        self.pseudotime_col = pseudotime_col
        self.num_segments = num_segments
        self.overlap_fraction = overlap_fraction
        self.max_clusters = max_clusters
        self.df_scaled = None
        self.segment_ranges = []
        self.segmented_df = None
        self.segment_datasets = {}
        self.segments_indices = {}
        self.S_segments = {}
        self.S_final_segments = {}
        # use later as dictionary with segments as keys and data subsets as values based on cells per segment
        self.S_clustering_segments = {}
        self.cluster_results = {}
        self.cluster_indices = {}

    def pseudotime_scaling(self):

        # scale pseudotime in [0,1] and sort the dataset by increasing pseudotime value
        df = self.df_original.copy()
        min_val = df[self.pseudotime_col].min()
        max_val = df[self.pseudotime_col].max()
        df[self.pseudotime_col] = (df[self.pseudotime_col] - min_val) / (max_val - min_val)
        df = df.sort_values(by=self.pseudotime_col)
        self.df_scaled = df

        return df

    def segmentation_with_overlapping_cells(self):

        # segment pseudotime column into overlapping segments, return the segment ranges and add a new column in the dataset that shows in which segment(s) a cell belongs to

        df = self.df_scaled.copy()
        pst_values = df[self.pseudotime_col].values
        min_val, max_val = pst_values.min(), pst_values.max()
        segment_width = (max_val - min_val) / self.num_segments

        # pseudotime is scaled, step_size is 0.1 * (1-0.35) = 0.065 for the default settings
        step_size = segment_width * (1 - self.overlap_fraction)

        self.segment_ranges = []
        for i in range(self.num_segments):
            start = min_val + i * step_size
            end = start + segment_width
            # set "end" equal to maximum pseudotime value if it is greater than that value
            if end > max_val:
                end = max_val
            self.segment_ranges.append((start, end))

        self.segment_ranges[-1] = (self.segment_ranges[-1][0], max_val)

        def segment_labeling(value):

            # label values by their pseudotime segment
            return [i for i, (start, end) in enumerate(self.segment_ranges) if start <= value <= end]

        df['Segments'] = df[self.pseudotime_col].apply(segment_labeling)
        self.segmented_df = df

        return df, self.segment_ranges

    def segmented_dictionaries_creation(self):

        # dictionary with keys as segments and values as a subset of the original dataset
        df = self.segmented_df.copy()
        segment_col = 'Segments'
        seg_indices = sorted(set(seg for sublist in df[segment_col] for seg in sublist))

        # unique segment indices
        for seg in seg_indices:
            seg_df = df[df[segment_col].apply(lambda x: seg in x)].copy()
            seg_df.drop(columns=[segment_col], inplace=True)
            self.segment_datasets[seg] = seg_df

        return self.segment_datasets

    def clustering_dictionaries_creation(self):

        # return the segmented datasets in a dictionary: with original cell indices and with/without pseudotime column to use later for clustering
        # extract all numeric columns except for pseudotime
        numeric_columns_wo_pst = [
            col for col in self.df_scaled.select_dtypes(include='number').columns
            if col != self.pseudotime_col
        ]

        # loop over each segment and its data
        for seg, data in self.segment_datasets.items():
            self.segments_indices[seg] = data.index.tolist()
            # store scaled data for segment in loop
            self.S_segments[f'S{seg}_'] = self.df_scaled.loc[self.segments_indices[seg]]
            # exclude pseudotime column from datasets
            final_data = self.S_segments[f'S{seg}_'][numeric_columns_wo_pst].copy()
            # add a new column to store the original indices of rows
            final_data['original_index'] = final_data.index
            self.S_final_segments[f'S{seg}'] = final_data
            # data for clustering, drop indices
            self.S_clustering_segments[f'S{seg}_for_clustering'] = final_data.drop(columns=['original_index'])

        return (
            self.segments_indices,
            self.S_segments,
            self.S_final_segments,
            self.S_clustering_segments
        )

    def gaussian_mixture_clustering_per_segment(self, random_state=1):

        # cluster cells with Gaussian mixture model (GMM) within each segment using BIC criterion (pseudotime column is not used during clustering)
        # return attributes for the clusters at segment: max number of clusters per segment, feature means and covariance matrix (full) per cluster, and cell indices assigned at each cluster

        num_segments = len(self.S_clustering_segments)

        for i in range(num_segments):
            X_data = self.S_clustering_segments[f'S{i}_for_clustering']
            best_GMM = None
            lowest_bic = np.inf

            for n in range(1, self.max_clusters + 1):
                GMM = GaussianMixture(n_components=n, random_state=1, covariance_type='full')
                GMM.fit(X_data)
                bic = GMM.bic(X_data)
                # update best GMM if GMM decreases
                if bic < lowest_bic:
                    best_GMM = GMM
                    lowest_bic = bic

            # predictions/cluster labels for cells falling into the segment in loop
            cluster_labels = best_GMM.predict(X_data)
            self.S_final_segments[f'S{i}']['cluster'] = cluster_labels

            # store in a dictionary the mean and the covariance matrix of each Gaussian component
            cluster_mean_cov = {
                cluster: {
                    'mean': best_GMM.means_[cluster],
                    'covariance': best_GMM.covariances_[cluster]
                } for cluster in range(best_GMM.n_components)
            }

            self.cluster_results[f'S{i}'] = {
                'best_GMM_model': best_GMM,
                'cluster_labels': cluster_labels,
                'cluster_means_covariance_mat': cluster_mean_cov
            }

            # for each unique cluster locate the original indices of its data points
            self.cluster_indices[f'S{i}'] = {
                cluster: self.S_final_segments[f'S{i}'].loc[
                    self.S_final_segments[f'S{i}']['cluster'] == cluster, 'original_index'].tolist()
                for cluster in np.unique(cluster_labels)
            }

        return self.cluster_results, self.cluster_indices

    def cell_states_creation_full_workflow(self, random_state=1):

        # run full workflow to create the cell states across pseudotime segments
        df_scaled = self.pseudotime_scaling()
        df_segmented, segment_ranges = self.segmentation_with_overlapping_cells()
        segment_datasets = self.segmented_dictionaries_creation()

        (
        segments_indices, S_segments, S_final_segments, S_clustering_segments) = self.clustering_dictionaries_creation()
        cluster_results, cluster_indices = self.gaussian_mixture_clustering_per_segment(random_state=1)

        return {
            "df_scaled": df_scaled,
            "df_segmented": df_segmented,
            "segment_ranges": segment_ranges,
            "segment_datasets": segment_datasets,
            "segments_indices": segments_indices,
            "S_segments": S_segments,
            "S_final_segments": S_final_segments,
            "S_clustering_segments": S_clustering_segments,
            "cluster_indices": cluster_indices,
            "cluster_results": cluster_results
        }

    def cluster_paths_backtracking(self, S_final_segments):

        # return a dictionary with connections between clusters at consecutive segments
        # each cluster connects to its closest one in the previous segment
        # wasserstein-2 distances are computed in the utils.py

        cluster_connections = {}
        backtracked_paths = {}

        # create cluster connections between consecutive segments
        # loop over all segments (except for S0)
        for seg in range(1, self.num_segments):
            # keys for current and previous segments
            current_segment = f"S{seg}"
            previous_segment = f"S{seg - 1}"
            # connections dictionary
            cluster_connections[current_segment] = {}

            # get results for both segments
            current_clusters = S_final_segments[current_segment]
            previous_clusters = S_final_segments[previous_segment]

            # loop over each unique cluster in current segment
            for current_cluster in current_clusters['cluster'].unique():
                # extract all cell indices for cluster in loop
                current_indices = set(
                    current_clusters.loc[current_clusters['cluster'] == current_cluster, 'original_index'])

                # count cells overlapping between current cluster and any unique cluster in previous segment (using the length of intersection of their cell indices)
                overlaps = {
                    previous_cluster: len(current_indices & set(
                        previous_clusters.loc[previous_clusters['cluster'] == previous_cluster, 'original_index'])
                                          )
                    for previous_cluster in previous_clusters['cluster'].unique()
                }

                # get the previous cluster with max overlap
                closest_previous_cluster = max(overlaps, key=overlaps.get)
                # store connection and number of overlapping cells
                cluster_connections[current_segment][current_cluster] = {
                    "connected_to": closest_previous_cluster,
                    "overlap_size": overlaps[closest_previous_cluster]
                }

        # backtrack cell state paths
        for seg in range(1, self.num_segments):
            # segment key
            current_segment = f"S{seg}"

            # loop over clusters in cluster connections of the segment in loop
            for current_cluster in cluster_connections[current_segment]:
                # initialize path to segment and cluster in loop (as a list of tuples)
                path = [(current_segment, current_cluster)]
                segment = current_segment

                while segment in cluster_connections:
                    segment_num = int(re.search(r'\d+', segment).group())
                    # get previous segment id
                    previous_segment_index = segment_num - 1
                    # invalid segment ids
                    if previous_segment_index < 0:
                        break

                    previous_segment = f"S{previous_segment_index}"
                    # get cluster from previous segment connected to current cluster
                    previous_cluster = cluster_connections[segment][path[-1][1]]["connected_to"]
                    path.append((previous_segment, previous_cluster))
                    segment = previous_segment

                # reverse the path from earliest to latest segments
                path.reverse()

                # build a string representation for the path
                path_str_parts = []

                # loop over each segment/cluster combination in the path
                for seg_name, cluster in path:
                    # extract ids
                    segment_num = int(re.search(r'\d+', seg_name).group())
                    # show connections between tuples with arrows, add also segment/cluster descriptor
                    path_str_parts.append(f"segment {segment_num}, cluster {cluster}")
                path_str = " -> ".join(path_str_parts)

                # store the path of each cluster in every segment as cluster_c_segment_S
                backtracked_paths[f"cluster_{current_cluster}_segment_{current_segment}"] = path_str

        return cluster_connections, backtracked_paths


