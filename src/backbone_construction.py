import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import itertools
import re


class InitialBackboneConstruction:

    # construct the initial trajectory backbone by merging paths terminating in the final segment

    def __init__(self, subset_costs):

        # intra-segment W-2 costs for paths terminating in final segment
        self.subset_costs = subset_costs
        # squared pairwise cost matrix (used later for path grouping)
        self.distance_matrix, self.all_clusters = self.costs_distance_matrix()

    def costs_distance_matrix(self):

        clusters = sorted(set(itertools.chain(*self.subset_costs.keys())))
        n = len(clusters)
        cluster_indices = {cluster: i for i, cluster in enumerate(clusters)}

        # initialize matrix for costs
        init_val = max([max(0, d) for d in self.subset_costs.values()]) + 1
        matrix = np.full((n, n), fill_value=init_val)

        for (a, b), dist in self.subset_costs.items():
            dist = max(0, dist)
            i, j = cluster_indices[a], cluster_indices[b]
            # symmetric costs
            matrix[i, j] = dist
            matrix[j, i] = dist

        np.fill_diagonal(matrix, 0)

        return matrix, clusters

    def label_assignment(self, groups, ungrouped):

        # label each group of paths to use for Silhouette score self-evaluation
        # ungrouped: clusters in final segment that were not merged with others

        cluster_labels = {}
        label = 0

        for group in groups:
            # loop over each unique cluster in a group and assign a cluster label id
            for cluster in group:
                cluster_labels[cluster] = label
            label += 1

        # label the ungrouped clusters with a unique label
        for cluster in ungrouped:
            cluster_labels[cluster] = label
            label += 1

        labels = [cluster_labels[cluster] for cluster in self.all_clusters]

        return np.array(labels)

    def path_group_merging(self, threshold):

        # merge clusters into groups (and thus their associated paths) if their pairwise cost is below optimal threshold
        groups = []
        # all unique clusters
        all_clusters = set(cluster for pair in self.subset_costs.keys() for cluster in pair)
        clusters_in_groups = set()
        # for each pair of clusters
        for (cluster1, cluster2), distance in self.subset_costs.items():

            if distance < threshold:
                # placeholder for clusters in groups
                group_found = False

                for group in groups:
                    if cluster1 in group or cluster2 in group:
                        # check in any of the clusters is in a group, if so merge them
                        # transitivity does not apply
                        # merge clusters into same group
                        group.add(cluster1)
                        group.add(cluster2)
                        clusters_in_groups.update([cluster1, cluster2])
                        # update their status to show they have been assigned to a group
                        group_found = True
                        break

                # make a new group of the clusters if none is already assigned to a group
                if not group_found:
                    new_group = {cluster1, cluster2}
                    groups.append(new_group)
                    clusters_in_groups.update([cluster1, cluster2])

        # find clusters not assigned to a group
        ungrouped_clusters = all_clusters - clusters_in_groups

        return groups, ungrouped_clusters

    def silhouette_threshold_selection(self, min_threshold=0.01, max_threshold=1.0, num_points=50):

        # create candidate thresholds
        thresholds = np.linspace(min_threshold, max_threshold, num_points)
        silhouette_scores = []

        for threshold in thresholds:
            # test the above merging function at each threshold
            groups, ungrouped = self.path_group_merging(threshold)
            # label clusters based on their groups (integers)
            labels = self.label_assignment(groups, ungrouped)

            # total groups
            n_labels = len(set(labels))
            # total labelled samples
            n_samples = len(labels)

            # skip if groups are less than 2 or >= to number of samples
            if n_labels < 2 or n_labels >= n_samples:
                silhouette_scores.append(-1)
                continue

            # for distance array: precomputed (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
            score = silhouette_score(self.distance_matrix, labels, metric="precomputed")
            silhouette_scores.append(score)

        # pick threshold with max silhouette score
        best_index = np.argmax(silhouette_scores)
        optimal_threshold = thresholds[best_index]

        return optimal_threshold

    def initial_backbone_output(self, groups, ungrouped_clusters, prefix="Group"):

        # merge in a dictionary both groups and ungrouped clusters (initial
        # trajectory backbone where keys are group ids and values in form
        # cluster_c_segment_s, here s is corresponds to the final segment

        group_dict = {}

        # add groups of pathways in dictionary along with an id
        for i, group in enumerate(groups):
            group_name = f"{prefix}_{i + 1}"
            group_dict[group_name] = list(group)

        # add in dictionary ungrouped clusters as separate pathways
        start_index = len(group_dict)
        for j, cluster in enumerate(sorted(ungrouped_clusters)):
            group_name = f"{prefix}_{start_index + j + 1}"
            group_dict[group_name] = [cluster]

        return group_dict

    def unique_cluster_segments_extraction_from_paths(self, all_pathway_groups, backtracked_paths):

        # take as input all pathway groups and find unique clusters appearing across all segments of the paths
        group_unique_cluster_segments = {}

        for group_name, keys in all_pathway_groups.items():

            # get backtracked paths keys
            values = {key: backtracked_paths[key] for key in keys if key in backtracked_paths}

            # reformat paths
            reformatted = {key: self.path_reformatting(val) for key, val in values.items()}

            unique_segments = set()
            for path in reformatted.values():
                unique_segments.update(path.split(" -> "))

            # unique clusters/segments pairs per group
            group_unique_cluster_segments[group_name] = sorted(unique_segments)

        return group_unique_cluster_segments

    def path_reformatting(self, value):

        # reformat paths with arrows to print connections
        segments = re.findall(r'segment (\d+), cluster (\d+)', value)

        return ' -> '.join([f'cluster_{cluster}_segment_S{segment}' for segment, cluster in segments])
