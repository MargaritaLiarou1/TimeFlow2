import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
import re
import scanpy
import flowsom as fs
import scanpy as sc

np.random.seed(1)


class CellStatePathConstructorFlowSOM:

    def __init__(self, df, pseudotime_col='pseudotime', num_segments=10, overlap_fraction=0.35, SOM_xdim=5, SOM_ydim=5):

        # dataset must contain CD markers and a pseudotime column, estimated by TimeFlow 1 (https://github.com/MargaritaLiarou1/TimeFlow)
        self.df_original = df.copy()
        self.pseudotime_col = pseudotime_col
        self.num_segments = num_segments
        self.overlap_fraction = overlap_fraction
        self.df_scaled = None
        self.segment_ranges = []
        self.segmented_df = None
        self.segment_datasets = {}
        self.segments_indices = {}
        self.S_segments = {}
        self.S_final_segments = {}
        # use later as dictionary with segments as keys and data subsets as values
        # based on values falling into a segment
        self.S_clustering_segments = {}
        self.cluster_results = {}
        self.cluster_indices = {}
        self.SOM_xdim = SOM_xdim
        self.SOM_ydim = SOM_ydim

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

        # segment pseudotime column into overlapping segments, return the segment ranges
        # and add a new column in the dataset that shows in which segment(s) a cell belongs to

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
            # set end equal to maximum pseudotime value if it is greater than that value
            if end > max_val:
                end = max_val
            self.segment_ranges.append((start, end))

        self.segment_ranges[-1] = (self.segment_ranges[-1][0], max_val)

        def segment_labeling(value):

            # labels values by their pseudotime segment
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

        # return a dictionaries with segmented datasets: with original cell indices and
        # with/without pseudotime to be used for clustering

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
            # data for clustering, do not include column with integer indices
            self.S_clustering_segments[f'S{seg}_for_clustering'] = final_data.drop(columns=['original_index'])

        return (
            self.segments_indices,
            self.S_segments,
            self.S_final_segments,
            self.S_clustering_segments
        )

    def flowsom_clustering_per_segment(self, random_state=42):

        # cluster cells with FlowSOM using a SOM grid
        # pseudotime column is not used during clustering
        # return all relevant results for the clusters at segment level

        num_segments = len(self.S_clustering_segments)

        # loop over segments
        for i in range(num_segments):
            X_data = self.S_clustering_segments[f'S{i}_for_clustering']
            # run FlowSOM with pandas dataset converted to AnnData object
            X_data_anndata = sc.AnnData(X_data)

            # select all numeric features/markers except for pseudotime column to cluster the cells
            cols_to_use = self.df_original.select_dtypes(include='number').columns.drop('pseudotime').tolist()

            # n_clusters for meta-clustering is not used in this manuscript
            fsom = fs.FlowSOM(X_data_anndata.copy(), cols_to_use=cols_to_use, n_clusters=2, xdim=self.SOM_xdim,
                              ydim=self.SOM_ydim, seed=42)

            # n_clusters for meta-clustering is not used in this manuscript
            ff_clustered = fs.flowsom_clustering(X_data_anndata, cols_to_use, xdim=self.SOM_xdim, ydim=self.SOM_ydim,
                                                 n_clusters=2, seed=42)

            cluster_labels = fsom.get_cell_data().obs.clustering
            # retrieve cluster labels
            cluster_labels = cluster_labels.to_numpy(dtype=int)

            self.S_final_segments[f'S{i}']['cluster'] = cluster_labels

            # unlike GMM there is no need to store means and covariance matrix for the clusters
            self.cluster_results[f'S{i}'] = {
                'cluster_labels': cluster_labels
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
        cluster_results, cluster_indices = self.flowsom_clustering_per_segment(random_state=42)

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

        # returns a dictionary with connections between clusters at consecutive segments
        # each cluster connects to its closest one in the previous segment
        # wasserstein distances are computed in the utils.py

        cluster_connections = {}
        backtracked_paths = {}

        # create cluster connections between consecutive segments
        # loop over all segments starting (except for S0)
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

                # count cell overlaps between current cluster and any unique cluster in previous segment
                # based on length of intersection of their cell indices
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
        # loop over segments
        for seg in range(1, self.num_segments):
            # extract segment key
            current_segment = f"S{seg}"

            # loop over clusters in cluster connections of segment in loop
            for current_cluster in cluster_connections[current_segment]:
                # initialize path to segment and cluster in loop (as a list of tuples)
                path = [(current_segment, current_cluster)]
                segment = current_segment

                while segment in cluster_connections:
                    segment_num = int(re.search(r'\d+', segment).group())
                    # get previous segment id
                    previous_segment_index = segment_num - 1
                    # break if the previous segment does not have a valid id
                    if previous_segment_index < 0:
                        break

                    previous_segment = f"S{previous_segment_index}"
                    # get cluster from previous segment connected to current cluster
                    previous_cluster = cluster_connections[segment][path[-1][1]]["connected_to"]
                    # append tuple to path
                    path.append((previous_segment, previous_cluster))
                    # proceed with previous segment in next loop
                    segment = previous_segment

                # reverse the path to go from earliest to latest
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

                # store the path of each cluster in every segment and fetch them based on cluster_c_segment_S
                backtracked_paths[f"cluster_{current_cluster}_segment_{current_segment}"] = path_str

        return cluster_connections, backtracked_paths




