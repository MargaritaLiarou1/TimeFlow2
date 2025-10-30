import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import itertools
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
import re
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr

np.random.seed(1)


def intra_segment_wasserstein_distances(cluster_results):
    # store in a nested dictionary the intra-segment Wasserstein-2 (W2) distances between the Gaussian components (cell states), manuscript Section 2.3.4
    # https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
    intra_segment_wasserstein_distances = {}

    # loop over all segments and clusters' metadata
    for segment, cluster_data in cluster_results.items():

        clusters = list(cluster_data["cluster_means_covariance_mat"].keys())
        wasserstein_distances_segment = {}

        for i, cluster_i in enumerate(clusters):

            mean_i = np.array(cluster_data["cluster_means_covariance_mat"][cluster_i]["mean"])
            covariance_i = np.array(cluster_data["cluster_means_covariance_mat"][cluster_i]["covariance"])

            for j, cluster_j in enumerate(clusters):
                # symmetric distances, skip double computation of pair i,j and j,i
                if i <= j:
                    mean_j = np.array(cluster_data["cluster_means_covariance_mat"][cluster_j]["mean"])
                    covariance_j = np.array(cluster_data["cluster_means_covariance_mat"][cluster_j]["covariance"])

                    # symmetric covariance
                    covariance_i = (covariance_i + covariance_i.T) / 2
                    covariance_j = (covariance_j + covariance_j.T) / 2

                    # sqrt(Sigma_2) * Sigma_1 * sqrt(Sigma_2)
                    sqrt_covariance_j = sqrtm(covariance_j)
                    covariances_term = sqrt_covariance_j @ covariance_i @ sqrt_covariance_j

                    # keep real values
                    sqrt_covariances_term = np.real_if_close(sqrtm(covariances_term))

                    # https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions
                    wasserstein_distance = (
                                np.linalg.norm(mean_i - mean_j) ** 2 + np.trace(covariance_i) + np.trace(covariance_j)
                                - 2 * np.trace(sqrt_covariances_term))

                    wasserstein_distances_segment[(i, j)] = wasserstein_distance

        intra_segment_wasserstein_distances[segment] = wasserstein_distances_segment

    return intra_segment_wasserstein_distances


def path_costs_computation(backtracked_paths, df_results):
    # compute pairwise costs of paths, manuscript section 2.3.4 equation 2
    def tuple_extraction(path_str):

        # find (segment,cluster) tuples
        pairs = re.findall(r"segment (\d+), cluster (\d+)", path_str)
        return [(int(seg), int(clust)) for seg, clust in pairs]

    def get_wasserstein_distances(segment, cluster1, cluster2):

        # get intra-segment W2 distances between two clusters in a given segment
        segment_key = f"S{segment}"
        cluster_pair = f"{min(cluster1, cluster2)},{max(cluster1, cluster2)}"
        row = df_results[
            (df_results["Segment"] == segment_key) & (df_results["Cluster_pair"] == cluster_pair)
            ]
        return row["Wasserstein_2_distance"].values[0]

        # paths with same length (= number of segments)

    paths_by_length = {}
    parsed_paths = {}

    # loop over keys (cluster_c_segment_S) and their paths in backtracked paths dictionary
    for key, path in backtracked_paths.items():
        # get tuple pairs
        tuple_pairs = tuple_extraction(path)
        parsed_paths[key] = tuple_pairs
        path_length = len(tuple_pairs)
        # paths with new length
        if path_length not in paths_by_length:
            paths_by_length[path_length] = []
        paths_by_length[path_length].append(key)

    # pairwise distances calculation
    path_costs = {}

    # loop over all paths by length
    for length, paths in paths_by_length.items():
        # loop over each pair of paths
        for path1, path2 in itertools.combinations(paths, 2):
            # parsed path1 and path2
            trans1 = parsed_paths[path1]
            trans2 = parsed_paths[path2]
            # set initial distance to 0
            total_distance = 0

            # see equation 2 in section 2.3.4 for accumulated W2 costs
            for (seg, clust1), (_, clust2) in zip(trans1, trans2):
                total_distance += get_wasserstein_distances(seg, clust1, clust2)

            path_costs[(path1, path2)] = total_distance

    return path_costs


def leaf_clusters_extraction(backtracked_paths):
    # count occurrences of each cluster_c_segment_s across all paths
    # return a list of clusters that appear only once in the backtracked paths and are not found within the final segment (leaf clusters)
    def path_transformation(path):
        transformed_clusters = []
        segments = path.split(" -> ")
        for seg in segments:
            match = re.search(r'segment (\d+), cluster (\d+)', seg)
            if match:
                seg_num, cluster_num = match.groups()
                transformed_clusters.append(f'cluster_{cluster_num}_segment_S{seg_num}')

        return " -> ".join(transformed_clusters)

    # keys: clusters per segment (cluster_c_segment_s), values: connected clusters
    transformed_paths = {key: path_transformation(value) for key, value in backtracked_paths.items()}
    transformed_paths_list = {key: value.replace(" -> ", ", ") for key, value in transformed_paths.items()}

    # flatten clusters from all segments in a list to count their occurences
    all_clusters = []
    for path in transformed_paths_list.values():
        all_clusters.extend(path.split(", "))

    # cluster occurrences
    cluster_counts = Counter(all_clusters)
    # get index of final segment
    final_segment = max(int(el.split("_S")[-1]) for el in all_clusters)
    # find clusters appearing only once, and are not found in the final segment
    leaf_clusters = [cluster for cluster, count in cluster_counts.items()
                     if count == 1 and int(cluster.split("_S")[-1]) < final_segment]

    return leaf_clusters


def top_N_cluster_search(leaf_clusters, intra_segment_wasserstein_distances, top_N=4):
    # make a dictionary where keys correspond to ordered top-1,...,top-N within segment and values to the corresponding closest cluster (based on intra-segment 2-wasserstein distances)
    # Section 2.3.5

    segment_clusters = defaultdict(set)

    # maps clusters per segment
    for entry in leaf_clusters:
        match = re.match(r"cluster_(\d+)_segment_S(\d+)", entry)
        if match:
            cluster_id = int(match.group(1))
            segment = f"S{int(match.group(2))}"
            segment_clusters[segment].add(cluster_id)

    top_N_clusters = {f"top_{i + 1}": {} for i in range(top_N)}

    for seg, leaf_cluster in segment_clusters.items():
        dist_map = intra_segment_wasserstein_distances.get(seg, {})
        all_clusters = set()
        for i, j in dist_map.keys():
            all_clusters.update([i, j])

        if not all_clusters:
            continue

        for top in top_N_clusters:
            top_N_clusters[top][seg] = {}

        for i in leaf_cluster:

            distances = []

            for j in all_clusters:
                if i == j:
                    continue
                d = dist_map.get((i, j), dist_map.get((j, i)))
                if d is not None:
                    distances.append((j, d))
            # sort by increasing distance
            distances.sort(key=lambda x: x[1])

            # find clusters at each N-th level
            at_top_N_dist_clusters = {}
            explored_distances = []

            for j, d in distances:
                if d not in explored_distances:
                    if len(explored_distances) >= top_N:
                        break
                    explored_distances.append(d)
                    at_top_N_dist_clusters[d] = [j]
                else:
                    at_top_N_dist_clusters[d].append(j)

            # save in a nested dictionary
            for index, (d_val, cluster_ids) in enumerate(at_top_N_dist_clusters.items()):
                top_key = f"top_{index + 1}"
                # list for closest clusters
                top_N_clusters[top_key][seg][f"cluster_{i}"] = [
                    f"cluster_{j}" for j in sorted(set(cluster_ids))
                ]

    return top_N_clusters


def final_segment_path_costs(path_costs):
    # return path costs for paths terminating in final segment
    all_segments = {
        re.search(r'segment_(S\d+)', part).group(1)
        for key in path_costs
        for part in key
        if re.search(r'segment_(S\d+)', part)
    }

    # find final segment
    final_segment = max(all_segments, key=lambda x: int(x[1:]))

    # subset only paths costs relevant to paths in final segment
    subset_costs = {
        key: value
        for key, value in path_costs.items()
        if f'segment_{final_segment}' in key[0] or f'segment_{final_segment}' in key[1]
    }

    return final_segment, subset_costs


def parse_path(path_str):
    # split input by arrow to get steps of path (segment, cluster)
    steps = path_str.split(" -> ")
    sequence = []

    for step in steps:
        parts = step.split(", ")
        # extract integers for segment and cluster
        segment = int(parts[0].split(" ")[1])
        cluster = int(parts[1].split(" ")[1])
        # append tuple in sequence
        sequence.append((segment, cluster))

    return sequence


def build_path_time_series_numeric_only(paths, indices_cells, X):
    # see relevant functions in time_series_utils.py
    # keep only numeric columns from the dataframe
    X_numeric = X.select_dtypes(include=np.number).drop(columns=["pseudotime"])

    ts_data = {}

    for path_name, path_str in paths.items():
        sequence = parse_path(path_str)
        mean_vectors = []

        for segment_idx, cluster_idx in sequence:
            seg_key = f"S{segment_idx}"
            try:
                indices = indices_cells[seg_key][cluster_idx]
                if len(indices) == 0:
                    mean_vector = np.full(X_numeric.shape[1], np.nan)
                else:
                    mean_vector = X_numeric.iloc[indices].mean().values
            except KeyError:
                mean_vector = np.full(X_numeric.shape[1], np.nan)

            mean_vectors.append(mean_vector)

        ts_data[path_name] = mean_vectors

    return ts_data


def merge_and_renumber_groups(*group_dicts):
    # collect all cluster groups from all input dictionaries
    all_groups = []
    for group_dict in group_dicts:
        all_groups.extend(group_dict.values())

    # merged groups
    merged = {f"Group_{i + 1}": group for i, group in enumerate(all_groups)}
    return merged


def build_group_time_series_numeric_only(groups_dict, indices_dict, df):
    # see relevant functions in time_series_utils.py, exclude pseudotime column for time series paths
    df_numeric = df.select_dtypes(include=np.number).drop(columns=["pseudotime"])
    feature_count = df_numeric.shape[1]
    group_time_series = {}

    for group_name, cluster_entries in groups_dict.items():
        # group clusters by segment
        segment_cluster_map = defaultdict(list)
        for entry in cluster_entries:
            match = re.match(r"cluster_(\d+)_segment_S(\d+)", entry)
            if match:
                cluster_id = int(match.group(1))
                segment = int(match.group(2))
                segment_cluster_map[segment].append(cluster_id)

        # sort segments and compute aggregated feature/marker vectors
        ts = []
        for segment in sorted(segment_cluster_map.keys()):
            segment_key = f"S{segment}"
            clusters = segment_cluster_map[segment]

            all_indices = []
            for cluster_id in clusters:
                indices = indices_dict.get(segment_key, {}).get(cluster_id, [])
                all_indices.extend(indices)

            if all_indices:
                mean_vector = df_numeric.iloc[all_indices].mean().values
            else:
                mean_vector = np.full(feature_count, np.nan)

            ts.append(mean_vector)

        group_time_series[group_name] = ts

    return group_time_series


def path_correlation(path1, path2):
    path1 = np.array(path1)
    path2 = np.array(path2)

    # check for same shape
    if path1.shape != path2.shape:
        return None
        # flatten in 1D list feature values, total number: number of markers * number of segments in the path
    path1_flat = path1.flatten()
    path2_flat = path2.flatten()
    # feature values Pearson correlation
    corr, _ = pearsonr(path1_flat, path2_flat)

    return corr


def compute_cluster_correlation_similarity_percentile(result, cell_indices, time_series_per_path=None,
                                                      high_corr_guard=0.9, pearson_corr_threshold=0.85,
                                                      percentile_10_threshold=0.75):
    # separate between leaf clusters whose similarity is below and above the Pearson correlation threshold (fixed default value at 0.85)
    # decrease threshold to 0.75 if the 10th percentile of all correlation scores is above 0.85, pass if all individual correlation scores above 0.9

    similarities = {}
    high_corr_sources = {}
    low_corr_only_sources = {}

    # correlation scores across all segments
    all_corrs = []
    for segment, cluster_links in result.items():
        for source_cluster, target_clusters in cluster_links.items():
            for target_cluster in target_clusters:
                source_key = f"{source_cluster}_segment_{segment.replace('segment_', 'S')}"
                target_key = f"{target_cluster}_segment_{segment.replace('segment_', 'S')}"
                if source_key in time_series_per_path and target_key in time_series_per_path:
                    corr = path_correlation(time_series_per_path[source_key], time_series_per_path[target_key])
                    if corr is not None:
                        all_corrs.append(corr)

    # compute the 10th percentile of all correlations
    if not all_corrs:
        threshold = pearson_corr_threshold
    else:
        perc10 = np.percentile(all_corrs, 10)
        threshold = percentile_10_threshold if perc10 >= pearson_corr_threshold else pearson_corr_threshold
    print(f"Global 10th percentile = {perc10:.3f} → threshold = {threshold:.2f}")

    # distinguish between correlation scores below and above the threshold
    for segment, cluster_links in result.items():
        similarities[segment] = {}
        high_corr_sources[segment] = {}
        low_corr_only_sources[segment] = []

        print(f"\n Segment: {segment}")
        for source_cluster, target_clusters in cluster_links.items():
            similarities[segment][source_cluster] = {}
            above_count = 0

            for target_cluster in target_clusters:
                source_key = f"{source_cluster}_segment_{segment.replace('segment_', 'S')}"
                target_key = f"{target_cluster}_segment_{segment.replace('segment_', 'S')}"
                if source_key in time_series_per_path and target_key in time_series_per_path:
                    corr = path_correlation(time_series_per_path[source_key], time_series_per_path[target_key])
                    if corr is None:
                        continue

                    similarities[segment][source_cluster][target_cluster] = corr

                    # pass correlations above 0.9
                    if corr >= threshold or corr >= high_corr_guard:
                        high_corr_sources[segment].setdefault(source_cluster, []).append(target_cluster)
                        status = "Above threshold"
                        above_count += 1
                    else:
                        status = "Below threshold"

                    print(f"{source_key} - {target_key}: corr={corr:.3f} → {status}")

            if target_clusters and above_count == 0:
                low_corr_only_sources[segment].append(f"{source_cluster}_segment_{segment.replace('segment_', 'S')}")

    print("\n Source clusters with only below-threshold correlations:")

    for segment, clusters in low_corr_only_sources.items():
        print(f"{segment}: {clusters}")

    low_corr_only_sources = [item for sublist in low_corr_only_sources.values() for item in sublist]

    return similarities, high_corr_sources, threshold, low_corr_only_sources


def reformat_value(value):
    # format: cluster_c_segment_S
    segments = re.findall(r'segment (\d+), cluster (\d+)', value)
    return ' -> '.join([f'cluster_{cluster}_segment_S{segment}' for segment, cluster in segments])


def targets_per_source(top_N_results):
    # see manuscript Section 2.3.5 for top-N rule of explore clusters
    segment_source_targets = defaultdict(lambda: defaultdict(set))

    # loop over each value of top-N closest clusters and segments
    for top_n, segments in top_N_results.items():
        for segment, source_targets in segments.items():
            # target clusters of source clusters per segment
            for source, targets in source_targets.items():
                segment_source_targets[segment][source].update(targets)

    return segment_source_targets


def unexplored_sources_by_target(segment_source_targets, leaf_clusters):
    flat_per_segment = {}

    # loop over segments and their source_target elements
    for segment, source_targets in segment_source_targets.items():

        matching_sources = []
        # change format as in other functions
        segment_full = segment.replace('S', 'segment_S')

        # loop over each source leaf cluster and its targets
        for source, targets in source_targets.items():
            # reformat targets by adding updated segment key
            full_target_ids = [f"{target}_{segment_full}" for target in targets]
            # if all of target ids exist in the leaf clusters list add source to the matching list
            if all(full_id in leaf_clusters for full_id in full_target_ids):
                matching_sources.append(source)

        # store matching clusters for segment in loop
        if matching_sources:
            flat_per_segment[segment] = sorted(matching_sources)

    return flat_per_segment


def format_source_list(flat_per_segment):
    # reformat the leaf clusters whose closest clusters are all unexplored with cluster_c_segment_S
    # combine later with leaf clusters wiith low correlation and leaf clusters for DTW absorption
    formatted_source_list = []

    for segment, clusters in flat_per_segment.items():
        segment_x = segment.replace("segment_", "S")

        for cluster in clusters:
            formatted_id = f"{cluster}_segment_{segment_x}"
            formatted_source_list.append(formatted_id)

    return formatted_source_list


def leaf_clusters_groups(formatted_sources, low_corr_sources, leaf_clusters):
    # combine leaf clusters with low correlated top-1 closest cluster, leaf clusters whose closest clusters are all leaf clusters and leaf clusters to be absorbed by DTW
    # for the first two cases we consider the leaf clusters' paths as new pathway candidates in the trajectory backbone

    combined_sources = formatted_sources + low_corr_sources
    dtw_elements = [item for item in leaf_clusters if item not in combined_sources]

    grouped_dict = {
        f"Group_{i + 1}": [source]
        for i, source in enumerate(combined_sources)
    }

    return grouped_dict, dtw_elements


def extract_unique_segments_per_group(grouped_dict, backtracked_paths, reformat_func):
    result = {}

    for group_name, keys in grouped_dict.items():
        # for keys in the backtracked_paths
        values = {key: backtracked_paths[key] for key in keys if key in backtracked_paths}

        reformatted = {key: reformat_func(val) for key, val in values.items()}

        # extract unique segments from the reformatted paths
        unique_segments = set()
        for path in reformatted.values():
            segments = path.split(" -> ")
            unique_segments.update(segments)

        result[group_name] = sorted(unique_segments)

    return result


def filter_dfs_by_groups(updated_main_no_duplicates, backtracked_paths, indices_cells, df):
    def reformat_value(value):
        segments = re.findall(r'segment (\d+), cluster (\d+)', value)
        return [f'cluster_{cluster}_segment_S{segment}' for segment, cluster in segments]

    refined_groups_dfs = {}

    for group_name, keys_to_fetch in updated_main_no_duplicates.items():
        unique_cluster_segments = set()
        unique_indices = set()

        # cluster/segment pairs
        for key in keys_to_fetch:
            path_value = backtracked_paths.get(key)
            if path_value:
                segments = reformat_value(path_value)
                unique_cluster_segments.update(segments)
            else:
                match = re.match(r'cluster_(\d+)_S0', key)
                if match:
                    cluster_number = int(match.group(1))
                    # past code only for first segment
                    if cluster_number in indices_cells.get("S0", {}):
                        unique_indices.update(indices_cells["S0"][cluster_number])

        # cell indices for all relevant cluster/segment pairs
        for cluster_segment in unique_cluster_segments:
            # extract cluster and segment ids
            cluster_part = cluster_segment.split('_')[1]
            segment_part = cluster_segment.split('_')[-1]
            # set to segment key
            segment_key = segment_part
            cluster_number = int(cluster_part)

            if segment_key in indices_cells and cluster_number in indices_cells[segment_key]:
                unique_indices.update(indices_cells[segment_key][cluster_number])

        # unique, sorted indices
        unique_indices_list = sorted(unique_indices)
        # drop duplicate rows
        filtered_df = df.loc[unique_indices_list].drop_duplicates()

        refined_groups_dfs[group_name] = filtered_df

        print(f"\n {group_name} ")
        print(filtered_df.shape)

    return refined_groups_dfs


def get_top_5pct_cells_labeled(data_dict, pseudotime_col='pseudotime'):
    dfs = []
    for group, df in data_dict.items():
        # compute the 95th quantile of the pseudotime distribution and set it as cutoff value
        cutoff = df[pseudotime_col].quantile(0.95)
        # extract only cells with pseudotime value above the cutoff as terminal cells
        top_df = df[df[pseudotime_col] >= cutoff].copy()
        top_df['original_group'] = group
        dfs.append(top_df)

    return pd.concat(dfs, ignore_index=True)


def summarize_and_cluster_groups(df, group_df, group_col='original_group', exclude_col='pseudotime',
                                 threshold_range=np.linspace(0.1, 10, 50), manual_threshold=None):
    # hierarchical agglomerative clustering for terminal cells based on their summary statistics
    # model evaluated at different thresholds (unless a user threshold is given)
    # do not include pseudotime column for summary stats computation

    numeric_cols = df.select_dtypes(include='number').columns.drop(exclude_col, errors='ignore')

    summary_stats = []

    for group, group_df in group_df.groupby(group_col):
        stats = {group_col: group}
        for feature in numeric_cols:
            values = group_df[feature].dropna()
            # descriptive stats
            stats[f'{feature}_mean'] = values.mean()
            stats[f'{feature}_median'] = values.median()
            stats[f'{feature}_q25'] = values.quantile(0.25)
            stats[f'{feature}_q75'] = values.quantile(0.75)
        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats).set_index(group_col)
    # scale the stats
    X_summary = StandardScaler().fit_transform(summary_df)

    # ward linkage matrix
    ward = linkage(X_summary, method='ward')

    # default of manual threshold set to None
    if manual_threshold is not None:
        print(f"New groups based on manual threshold: {manual_threshold}")
        # hierarchical clustering with manual thresholding
        final_clusters = fcluster(ward, t=manual_threshold, criterion='distance')
        summary_df['cluster'] = final_clusters
        return summary_df, None, manual_threshold

    # automatic selection threshold selection with maximum Silhouette score (use cluster numeric labels, not gated populations)
    silhouette_scores = []

    for t in threshold_range:
        clusters = fcluster(ward, t=t, criterion='distance')
        n_clusters = len(set(clusters))
        n_samples = len(clusters)
        # check if clusters are less than data samples for clustering
        if n_clusters > 1 and n_clusters < n_samples:
            score = silhouette_score(X_summary, clusters)
        else:
            score = np.nan
        silhouette_scores.append(score)

    # if NaN Silhouette scores return group pathways as they are
    if all(np.isnan(silhouette_scores)):

        print("No clusters were found. Pathway groups remain the same.")
        summary_df['cluster'] = range(1, len(summary_df) + 1)
        best_threshold = None
    else:
        # ignore NaNs
        best_index = np.nanargmax(silhouette_scores)
        best_threshold = threshold_range[best_index]
        best_score = silhouette_scores[best_index]
        final_clusters = fcluster(ward, t=best_threshold, criterion='distance')
        summary_df['cluster'] = final_clusters

        # print(f"Best distance threshold: {best_threshold:.2f}")
        # print(f"Best silhouette score: {best_score:.2f}")

    print("Cluster assignment for each group:")
    print(summary_df[['cluster']])

    return summary_df, silhouette_scores, best_threshold


def merge_groups_by_cluster(summary_df, refined_groups_dfs):

    # return dictionary with final, refined groups along with their subset datasets
    # drop index original group and make it a new column
    # print Celltype stats only for evaluation if a Celltype column with gates is available
    if summary_df.index.name == 'original_group':
        summary_df = summary_df.reset_index()

    cluster_to_groups = summary_df.groupby('cluster')['original_group'].apply(list).to_dict()
    merged_dict = {}
    # optional if Celltype column available for evaluating cell type proportions
    celltype_stats = {}

    for cluster, groups in cluster_to_groups.items():
        # concatenated datasets (by rows) for all groups in same cluster
        merged_df = pd.concat([refined_groups_dfs[group] for group in groups], axis=0)
        # drop duplicate rows
        merged_df = merged_df.drop_duplicates()
        # store the merged dataset
        merged_dict[cluster] = merged_df

        # calculate celltype counts and proportions
        # counts = merged_df['Celltype'].value_counts()
        # proportions = merged_df['Celltype'].value_counts(normalize=True)

        # celltype_stats[cluster] = pd.DataFrame({
        #    'Count': counts,
        #   'Proportion': proportions
        # })

    return merged_dict, celltype_stats
