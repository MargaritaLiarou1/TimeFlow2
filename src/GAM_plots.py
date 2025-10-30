import os
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s


def marker_evolution_visualization_and_lineage_extraction(refined_pathway_groups,
                                                          markers=['CD14', 'CD16', 'CD36', 'CD19'],
                                                          output_folder='./results/P1/lineages-csv-files/'):
    
    # change to any marker based on the dataset, prioritize lineage-specific markers (if available) to identify expected lineages
    # convert refined pathways into a dictionary
    refined_pathway_groups = refined_pathway_groups[0]

    os.makedirs(output_folder, exist_ok=True)

    groups = sorted(refined_pathway_groups.keys())
    n_groups = len(groups)
    n_markers = len(markers)

    fig, axes = plt.subplots(nrows=n_markers, ncols=n_groups, figsize=(3 * n_groups, 3 * n_markers), sharey='row')
    if n_groups == 1:
        axes = axes[:, None]
    elif n_markers == 1:
        axes = axes[None, :]

    for col, group in enumerate(groups):
        df = refined_pathway_groups[group].copy()
        df = df.dropna(subset=['pseudotime'] + markers)

        # scale pseudotime in [0,1]
        pt_min, pt_max = df['pseudotime'].min(), df['pseudotime'].max()
        df['pseudotime_scaled'] = (df['pseudotime'] - pt_min) / (pt_max - pt_min)

        # calculate 0th and 100th percentiles
        # optional change of percentiles to exclude outliers
        pt_0 = np.percentile(df['pseudotime_scaled'], 0)
        pt_100 = np.percentile(df['pseudotime_scaled'], 100)
        # filter changes results only in case of outlier exclusion
        df_fit = df[(df['pseudotime_scaled'] >= pt_0) & (df['pseudotime_scaled'] <= pt_100)].copy()

        for row, marker in enumerate(markers):
            ax = axes[row, col]
            # label subplots
            label = chr(65 + row * n_groups + col)
            ax.text(-0.15, 1.2, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

            ax.scatter(df['pseudotime_scaled'], df[marker], s=0.05, alpha=1, color='#D5D8DC')

            x_fit = df_fit['pseudotime_scaled'].values
            y_fit = df_fit[marker].values
            # GAM model https://pygam.readthedocs.io/en/latest/api/lineargam.html
            gam = LinearGAM(s(0, n_splines=10, spline_order=3)).fit(x_fit, y_fit)
            # curve predictions
            x_grid = np.linspace(pt_0, pt_100, 100)
            y_pred = gam.predict(x_grid)
            ci = gam.confidence_intervals(x_grid)

            ax.plot(x_grid, y_pred, color='#0B3D91', linewidth=2)
            ax.fill_between(x_grid, ci[:, 0], ci[:, 1], color='cyan', alpha=0.2)

            ax.set_ylim(-0.1, 1)
            ax.set_xlabel('pseudotime', fontsize=15)
            if col == 0:
                ax.set_ylabel(marker, fontsize=15)
            if row == 0:
                ax.set_title(f'Pathway {group}', fontsize=15)
            ax.grid(False)
            if row == 0 and col == n_groups - 1:
                ax.legend(loc='best', frameon=False)

        # save each lineage as a separate CSV file for further analysis
        csv_path = os.path.join(output_folder, f'lineage_dataset_without_celltypes_indices_{col + 1}.csv')
        df.to_csv(csv_path, index=False)

    plt.tight_layout()
    plt.show()