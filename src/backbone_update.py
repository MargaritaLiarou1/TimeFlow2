import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

np.random.seed(1)


class TimeSeriesDTWMatch:

    def __init__(self, unmatched_ts, main_ts, unmatched_clusters, main_clusters):

        # Section 2.3.5, Trajectory Refinement to absorb leaf clusters by the trajectory backbone
        # get the data for each cluster's path that will be absorbed and use their time series representation, same for main backbone pathways
        # merge based on DTW distance: https://cs.fit.edu/~pkc/papers/tdm04.pdf
        # https://pypi.org/project/fastdtw/
        self.unmatched_ts = unmatched_ts
        self.main_ts = main_ts
        self.unmatched_clusters = unmatched_clusters
        self.main_clusters = main_clusters

    def timeseries_preprocessing(self, ts):

        # remove NaN from ts features (ts learn: to_time_series_dataset pads unequal length time series)
        mask = ~np.isnan(ts).any(axis=1)
        return ts[mask]

    def timeseries_dtw_matching(self):

        # match each unmatched leaf cluster (represented by a time series) to the closest backbone pathway (after updating it with possible new pathways that terminate earlier)
        # minimal DTW distance to identify closest pathway

        matches = []

        # loop over each time series
        for i in tqdm(range(self.unmatched_ts.shape[0]), desc="Absorbing time series with DTW"):
            # check for NaN
            un_ts = self.timeseries_preprocessing(self.unmatched_ts[i])
            best_index = None
            best_dist = float('inf')

            for j in range(self.main_ts.shape[0]):

                main_ts_clean = self.timeseries_preprocessing(self.main_ts[j])
                # https://pypi.org/project/fastdtw/
                dist, _ = fastdtw(un_ts, main_ts_clean, dist=euclidean)
                # update distance only only for lower distances (lower DTW, closer pathways)
                if dist < best_dist:
                    best_dist = dist
                    best_index = j

            matches.append((i, best_index, best_dist))

            print(
                f"Unmatched[{i}] leaf cluster matches best with backbone Main[{best_index}] at DTW distance {best_dist:.2f}")

        return matches

    def matching_dictionary(self, matches):

        # mappings between unmatched leaf clusters and main pathway groups
        matches_dict = {}
        for unmatched_index, main_index, _ in matches:
            unmatched_group = f"Group_{unmatched_index + 1}"
            main_group = f"Group_{main_index + 1}"
            matches_dict[unmatched_group] = main_group

        return matches_dict

    def backbone_absorption(self, matches_dict):

        # update main backbone
        updated_main = self.main_clusters.copy()

        for unmatched_group, matched_main_group in matches_dict.items():
            if unmatched_group in self.unmatched_clusters:
                unmatched_segments = self.unmatched_clusters[unmatched_group]
                updated_main.setdefault(matched_main_group, []).extend(unmatched_segments)

        return updated_main

    def duplicate_removal(self, cluster_dict):

        # remove duplicate cluster_c_segment_s from the groups based on their paths
        for group_name, cluster_list in cluster_dict.items():
            cluster_dict[group_name] = list(set(cluster_list))

        return cluster_dict

    def backbone_update_full_workflow(self):

        # call the previous functions and remove duplicate paths in a group
        matches = self.timeseries_dtw_matching()
        matches_dict = self.matching_dictionary(matches)
        merged_main = self.backbone_absorption(matches_dict)
        cleaned_main = self.duplicate_removal(merged_main)

        return cleaned_main
