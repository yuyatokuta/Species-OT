import os
import csv
import math
import statistics
import functools
import warnings
import logging
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

import anndata
import networkx as nx
import screcode

from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.signal import argrelextrema
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, ttest_ind
from statsmodels.stats.multitest import multipletests

import sklearn.decomposition
import sklearn.manifold

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein


def configure_platform(platform="metal"):
    """Configure JAX backend (gpu, metal, or cpu).

    Select a platform (Apple Silicon, NVIDIA GPU, or other) and configure the JAX backend accordingly.
    
    Args:
        platform (str): The desired platform. Options are 'gpu', 'metal', or 'cpu'.
        Defaults to "metal".

    Examples:
        >>> configure_platform("cpu")
    """
    platform_map = {"metal": "METAL", "gpu": "gpu", "cpu": "cpu"}
    if platform.lower() not in platform_map:
        raise ValueError("Invalid platform. Choose 'gpu', 'metal', or 'cpu'.")
    
    jax.config.update("jax_platform_name", platform_map[platform.lower()])
    print(f"JAX is configured to use: {platform_map[platform.lower()]}")


def _calculate_intersections(species_list, transcription_factors, adata):
    """Return genes common to all species within transcription_factors."""
    intersections = []
    for spe in species_list:
        intersect = np.intersect1d(transcription_factors, adata[spe].var.index)
        print(spe, intersect.shape)
        intersections.append(intersect)
    return functools.reduce(np.intersect1d, intersections)


def _generate_list(A, B, n):
    """Return a list with A followed by n copies of B."""
    return [A] + [B] * n


def _format_gene_name(gene_name, format_type):
    """Format gene name according to the specified style."""
    if format_type == "all_capital_italic":
        return f"$\\it{{{gene_name.upper()}}}$"
    elif format_type == "capitalized_italic":
        return f"$\\it{{{gene_name.capitalize()}}}$"
    return gene_name


def _plot_gene_expression(
    species,
    species_labels,
    species_name,
    gene_name,
    cells,
    plot_data,
    data_option,
    ax,
    title_fontsize=16,
    show_xlabel=False,
    show_ylabel=False,
    format_type="capitalized_italic",
):
    """Plot gene expression for a given species and gene."""
    ax.plot(plot_data[species_name][gene_name], marker="o")

    spe_dict = dict(zip(species, species_labels))

    formatted_gene_name = _format_gene_name(gene_name, format_type)
    ax.set_title(f"{spe_dict[species_name]} {formatted_gene_name}", fontsize=title_fontsize)

    num_cells = len(cells[species_name])

    if num_cells <= 12:
        xticks = range(num_cells)
        xticklabels = cells[species_name]
    else:
        num_labels = 12
        indices = np.linspace(0, num_cells - 1, num_labels, dtype=int)
        xticks = indices
        xticklabels = [cells[species_name][i] for i in indices]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")

    # Set labels based on the data option
    ylabel = "log-normalized expression level"
    xlabel = ""
    if data_option == "dataset2":
        ylabel = "log2(RP100K+1)"
        xlabel = "Single-cells"
        ax.set_ylim(0, 8)
    if data_option == "dataset1":
        ylabel = "log2(RPM+1)"
        xlabel = "Sampling timing"
        ax.set_ylim(0, 14)

    if show_ylabel:
        ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(xlabel)


def _plot_gene_expression_dataset1(
    species,
    species_labels,
    species_name,
    gene_name,
    cells,
    plot_data,
    data_option,
    ax,
    title_fontsize=16,
    show_xlabel=False,
    show_ylabel=False,
    format_type="capitalized_italic",
):
    """Plot dataset1 gene expression as time-series for a given species and gene."""
    y_values = plot_data[species_name][gene_name]
    num_points = len(y_values)
    num_timepoints = num_points // 2
    
    x_values_1 = np.arange(num_timepoints)
    x_values_2 = np.arange(num_timepoints) 
    
    mean_values = [
        np.mean(y_values[i * 2 : i * 2 + 2]) for i in range(num_timepoints)
    ]
    
    ax.scatter(x_values_1, y_values[::2], color="blue", marker="o", label="Data #1")
    ax.scatter(x_values_2, y_values[1::2], color="green", marker="o", label="Data #2")
    
    ax.plot(x_values_1, mean_values, marker="s", color="red", linestyle="--", label="Mean")

    spe_dict = dict(zip(species, species_labels))
    
    formatted_gene_name = _format_gene_name(gene_name, format_type)
    ax.set_title(f"{spe_dict[species_name]} {formatted_gene_name}", fontsize=title_fontsize)
    
    if species_name == "mouse":
        clean_labels = [label.split("#")[0] for label in cells[species_name][::2]]
    else:
        clean_labels = [label.split("1")[0] for label in cells[species_name][::2]]
    ax.set_xticks(x_values_1)
    ax.set_xticklabels(clean_labels, rotation=30, ha="right")
    
    ylabel = "log2(RPM+1)"
    xlabel = "Sampling timing"
    ax.set_ylim(0, 14)
    
    if show_ylabel:
        ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(xlabel)
    
    ax.legend()


def assign_colors(labels, target_labels, default_color):
    """Assign colors to labels based on target label membership."""
    return [
        "blue" if label in target_labels else default_color
        for label in labels
    ]


def _calculate_shared_genes(species_list, hvlabel_):
    """Return genes shared across all species."""
    intersections = []
    for spe in species_list:
        masked_genes = hvlabel_[spe]
        intersections.append(masked_genes)
    shared_genes = functools.reduce(np.intersect1d, intersections)
    return shared_genes


def _weighted_cdist(X, w=None):
    """Compute weighted pairwise Euclidean distances."""
    X = np.asarray(X)

    if w is None:
        w = np.ones(X.shape[1])
    else:
        w = np.asarray(w)

    # Validate dimensions
    if w.shape[0] != X.shape[1]:
        raise ValueError(
            "Weights array must match the dimensionality of the observations"
        )

    # Compute the squared weighted differences
    m, n = X.shape
    distances_squared = np.empty((m, m))
    for i in range(m):
        for j in range(m):  # Iterate over X again for the second set of points
            diff = X[i, :] - X[j, :]
            distances_squared[i, j] = np.dot(w * diff, diff)

    # Return the square root of the summed squared weighted differences (Euclidean distance)
    return np.sqrt(distances_squared)


def _compute_geometries(distance_matrices):
    """Create Geometry objects from distance matrices."""
    return [geometry.Geometry(cost_matrix=dm) for dm in distance_matrices]


def _print_convergence_results(epsilon, converged):
    """Print GWOT convergence status for a given epsilon."""
    status = "converged" if converged else "non-converged"
    print(f"epsilon = {epsilon:.7f} {status}")


def _check_convergence(epsilon, geometries, iterations):
    """Run GWOT for all species pairs and check convergence."""
    linear_solver = sinkhorn.Sinkhorn()

    all_converged = True
    coupling_matrices = {}
    gw_dis_mat = jnp.zeros((len(geometries), len(geometries)))

    for idx_i, (spec_i, geom_i) in enumerate(geometries.items()):
        for idx_j, (spec_j, geom_j) in enumerate(geometries.items()):
            problem = quadratic_problem.QuadraticProblem(
                geom_xx=geom_i, geom_yy=geom_j
            )
            gw_solver = gromov_wasserstein.GromovWasserstein(
                linear_solver=linear_solver, epsilon=epsilon, max_iterations=iterations
            )
            result = gw_solver(problem)

            n_converged_iterations = jnp.sum(result.costs != -1)
            # sinkhorn_converged = bool(
            #     result.linear_convergence[n_converged_iterations - 1]
            # )
            sinkhorn_converged = bool(result.linear_convergence[-1]) if len(result.linear_convergence) > 0 else False
            gw_converged = result.converged

            pair_converged = sinkhorn_converged and gw_converged
            all_converged &= pair_converged

            species_pair_key = f"{spec_i}_{spec_j}"
            coupling_matrices[species_pair_key] = result.matrix
            # gw_dis_mat = gw_dis_mat.at[idx_i, idx_j].set(result.reg_gw_cost)
            gw_dis_mat = gw_dis_mat.at[idx_i, idx_j].set(float(result.reg_gw_cost))
            
    return all_converged, coupling_matrices, gw_dis_mat


def _calculate_entropy(P):
    """Compute entropy H(P) = -Σ P_ij (log P_ij - 1)."""
    log_P = np.where(P > 0, np.log(P + 1e-12), 0)  # avoid log(0)
    return -np.sum(P * (log_P - 1))


def _append_to_returnable_object_type_list(raw_return_opt, returnable_object_type_list, object_type, object_name):
    """Append object type to return list if raw output is requested."""
    if raw_return_opt:
        object_discript_dict = {
            "heatmap": "pd.DataFrame",
            "dendrogram": "a linkage matrix (np.darray)",
            "jnp_heatmap": "jnp.darray"
        }
            
        returnable_object_type_list.append(object_discript_dict[object_type])
        print(f"The returned object (`{object_name}`) is {returnable_object_type_list[-1]}")

    return returnable_object_type_list


def _print_side_by_side(*args, width=20):
    """Print multiple text blocks side by side."""
    # Convert all arguments to strings
    str_args = [str(arg).split("\n") for arg in args]

    # Find the maximum number of lines in any argument
    max_lines = max(len(lines) for lines in str_args)

    # Pad all arguments to have the same number of lines
    for lines in str_args:
        lines.extend([""] * (max_lines - len(lines)))

    # Print each line of the arguments side by side
    for line_set in zip(*str_args):
        print("".join(f"{line:<{width}}" for line in line_set)) 


# def _merge_pairs_with_average(df):
#     """Return a DataFrame averaging two batches of samples."""
#     new_rows = []
#     new_index = []

#     # Take the average of two rows each. (2k, 2k+1)
#     # for i in range(0, len(df), 2):
#     for i in range(0, len(df) - 1, 2):
#         row1 = df.iloc[i]
#         row2 = df.iloc[i + 1]
#         avg_row = (row1 + row2) / 2.0

#         # Remove trailing “#1”, “#2”, “1”, “2” from labels
#         original_label = df.index[i]

#         if original_label.endswith("#1") or original_label.endswith("#2"):
#             label = original_label[:-2]  
#         elif original_label.endswith("1") or original_label.endswith("2"):
#             label = original_label[:-1]  
#         else:
#             label = original_label  

#         new_rows.append(avg_row)
#         new_index.append(label)

#     # Creat a new dataframe
#     merged_df = pd.DataFrame(new_rows, index=new_index)

#     return merged_df
def _merge_pairs_with_average(df):
    """Return a DataFrame averaging every two consecutive rows (2k, 2k+1).

    - If the number of rows is odd, no processing is performed and a warning is issued.
    - Row labels ending with '#1', '#2', '1', or '2' are unified by removing the suffix.
    """
    n = len(df)
    if n % 2 != 0:
        warnings.warn(
            f"_merge_pairs_with_average: DataFrame has an odd number of rows ({n}); no averaging performed.",
            UserWarning,
            stacklevel=2,
        )
        return df.copy()

    new_rows = []
    new_index = []

    for i in range(0, n, 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        avg_row = (row1 + row2) / 2.0

        original_label = str(df.index[i])

        # Remove trailing suffixes like "#1", "#2", "1", or "2"
        if original_label.endswith(("#1", "#2")):
            label = original_label[:-2]
        elif original_label.endswith(("1", "2")):
            label = original_label[:-1]
        else:
            label = original_label

        new_rows.append(avg_row)
        new_index.append(label)

    merged_df = pd.DataFrame(new_rows, index=new_index, columns=df.columns)
    return merged_df


class Config:
    """Container for SpeciesOT configuration parameters."""
    def __init__(self, data_option, mouse_option, gene_option, threshold_option, metric_option, dismat_option, gwot_option, mask_option, iterations, threshold_eps, low_epsilon, high_epsilon, threshold_tol, threshold, threshold_surer, species, species_pairs, species_labels):
        self.data_option = data_option
        self.mouse_option = mouse_option
        self.gene_option = gene_option
        self.threshold_option = threshold_option
        self.metric_option = metric_option
        self.dismat_option = dismat_option
        self.gwot_option = gwot_option
        self.mask_option = mask_option      
        self.iterations = iterations
        self.threshold_eps = threshold_eps
        self.low_epsilon = low_epsilon
        self.high_epsilon = high_epsilon
        self.threshold_tol = threshold_tol
        self.threshold = threshold
        self.threshold_surer = threshold_surer
        self.species = species
        self.species_labels = species_labels
        self.species_pairs = [f"{s1}_{s2}" for s1 in species for s2 in species]


class Data(Config):
    """Retrieve and store dataset parameters."""
    def __init__(self, config_object):
        self.data_option = config_object.data_option
        self.mouse_option = config_object.mouse_option
        self.gene_option = config_object.gene_option
        self.threshold_option = config_object.threshold_option
        self.metric_option = config_object.metric_option
        self.dismat_option = config_object.dismat_option
        self.gwot_option = config_object.gwot_option
        self.mask_option = config_object.mask_option
        self.iterations = config_object.iterations
        self.threshold_eps = config_object.threshold_eps
        self.low_epsilon = config_object.low_epsilon
        self.high_epsilon = config_object.high_epsilon
        self.threshold_tol = config_object.threshold_tol
        self.threshold = config_object.threshold
        self.threshold_surer = config_object.threshold_surer
        self.species = config_object.species
        self.species_labels = config_object.species_labels
        self.species_pairs = config_object.species_pairs    


    def _set_index_and_columns_names_to_none(self, df):
        """Remove index and column names from a DataFrame."""
        df.index.name = None
        df.columns.name = None
        return df


    def _create_adata(self, df_csv, spe):
        """Create AnnData sorted by gene name and filtered for nonzero expression."""
        sorted_genes = {}
        adata = {}
        gene_expression_sums = {}

        adata[spe] = anndata.AnnData(df_csv[spe])  
        adata[spe].obs_names = df_csv[spe].index
        adata[spe].var_names = df_csv[spe].columns
        sorted_genes[spe] = adata[spe].var.index.sort_values()
        adata[spe] = adata[spe][:, sorted_genes[spe]]
        # gene_expression_sums[spe] = np.array(adata[spe].X.sum(axis=0)).flatten()
        gene_expression_sums[spe] = np.asarray(adata[spe].X.sum(axis=0)).ravel()
        adata[spe] = adata[spe][:, gene_expression_sums[spe] > 0]  
        # print(spe, adata[spe].shape)

        return adata[spe]
    

    def read_csv(self, verbose=False):
        """Read input CSVs and initialize AnnData objects."""
        
        # Choose the input directory based on data type
        input_dir = "../data" if self.data_option in {"dataset1", "dataset2"} else "../custom"
        df_csv, self.adata = {}, {}

        for spe in self.species:
            # Determine file path and index column ----
            if self.data_option == "dataset1":
                path = f"{input_dir}/dataset1_{spe}.csv"
                index_col = "gene"  # dataset1 uses gene names as index
            elif self.data_option == "dataset2":
                path = f"{input_dir}/dataset2_{spe}.csv"
                index_col = "CellID"  # dataset2 uses cell IDs as index
            elif self.data_option == "custom":
                path = f"{input_dir}/{spe}.csv"
                index_col = "CellID"
            else:
                raise ValueError(f"Unrecognized data_option: {self.data_option}")

            # Check if the file exists before reading
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            # Read CSV and remove index/column names
            df = pd.read_csv(path, header="infer", index_col=index_col)
            df = self._set_index_and_columns_names_to_none(df)

            # Process dataset1
            if self.data_option == "dataset1":
                # Optionally drop specific mouse samples if requested
                if spe == "mouse" and self.mouse_option == "drop":
                    df = df.drop(["mESC#1", "mESC#2"], axis=1, errors="ignore")

                # Transpose
                df = df.T.astype(np.float32)
                df = self._set_index_and_columns_names_to_none(df)

            # Process dataset2
            elif self.data_option == "dataset2" and spe == "mouse":
                # For mouse data in dataset2, convert gene names to uppercase
                df.columns = df.columns.str.upper()

            # Convert to AnnData
            self.adata[spe] = self._create_adata({spe: df}, spe)
            df_csv[spe] = df

        # Set gene name index for all species
        for spe in self.species:
            self.adata[spe].var.index.name = "gene"

        return self


    def normalization(self):
        """Normalize expression data and reduce noise."""
        if (
            self.data_option == "dataset2" or self.data_option == "custom"
        ):
            size_factor = 1e5
            for spe in self.species:
                adata_copy = self.adata[spe].copy()
                recode = screcode.RECODE(version=2)
                adata_copy.obsm["RECODE"] = recode.fit_transform(
                    np.array(adata_copy.X.copy(), dtype="int32")
                )
                adata_copy.obsm["normalized_log"] = np.log2(
                    size_factor
                    * (
                        adata_copy.obsm["RECODE"].T / np.sum(adata_copy.obsm["RECODE"], axis=1)
                    ).T
                    + 1
                )
                self.adata[spe] = adata_copy

        elif (
            self.data_option == "dataset1"
        ):
            for spe in self.species:
                adata_copy = self.adata[spe].copy()
                adata_copy.obsm["normalized"] = adata_copy.X.copy()
                adata_copy.obsm["normalized_log"] = np.log2(adata_copy.obsm["normalized"] + 1)
                self.adata[spe] = adata_copy
                
        else:
            print(self.data_option)

        return self


    def _get_sorted_selected_gene_expression(self, gene_selection, spe):
        """Return sorted expression matrix and gene list for selected genes."""
        # Identify genes in gene_selection
        is_selected = np.isin(self.adata[spe].var.index, gene_selection)

        # Extract selected gene names
        selected_genes = self.adata[spe].var.index[is_selected]

        # Sort selected genes alphabetically and get their indices
        sorted_indices = np.argsort(selected_genes)

        # Apply selection and sorting to expression data
        normalized_log_select_adata = self.adata[spe].obsm["normalized_log"][
            :, is_selected
        ][:, sorted_indices]

        # Final sorted list of selected gene names
        sorted_selected_vars = selected_genes[sorted_indices]

        return normalized_log_select_adata, sorted_selected_vars


    def read_tf(self):
        """Read human transcription factor genes from TFs/hTF.txt."""
        df_TF = pd.read_csv("../TFs/hTF.txt", delimiter="\t", index_col="gene_id")
        df_TF.sort_values("gene_type", inplace=True)
        transcription_factors = df_TF["gene_type"].values

        # Select genes based on the gene_option
        if self.gene_option == "intersection":
            gene_selection = _calculate_intersections(self.species, transcription_factors, self.adata)
        elif self.gene_option == "distinct":
            gene_selection = transcription_factors

        self.sorted_selected_vars = {}
        for spe in self.species:
            (
                self.adata[spe].obsm["normalized_log_select"], 
                self.sorted_selected_vars[spe] 
            ) = self._get_sorted_selected_gene_expression(gene_selection, spe)

        return self


class SpeciesOT(Data):
    """Core class for geometric inference of cross-species transcriptomic correspondence using Gromov–Wasserstein optimal transport."""
    def __init__(self, data_object):
        self.data_option = data_object.data_option
        self.mouse_option = data_object.mouse_option
        self.gene_option = data_object.gene_option
        self.threshold_option = data_object.threshold_option
        self.metric_option = data_object.metric_option
        self.dismat_option = data_object.dismat_option
        self.gwot_option = data_object.gwot_option
        self.mask_option = data_object.mask_option
        self.iterations = data_object.iterations
        self.threshold_eps = data_object.threshold_eps
        self.low_epsilon = data_object.low_epsilon
        self.high_epsilon = data_object.high_epsilon
        self.threshold_tol = data_object.threshold_tol
        self.threshold = data_object.threshold
        self.threshold_surer = data_object.threshold_surer
        self.species = data_object.species
        self.species_labels = data_object.species_labels
        self.species_pairs = data_object.species_pairs  
        self.adata = data_object.adata
        self.sorted_selected_vars = data_object.sorted_selected_vars  


    def _obtain_maximum_and_median_gene_expression_level(self):
        """Compute per-gene maximum and median expression levels for each species."""
        maxvalue = {}
        medvalue = {}
        for spe in self.species:
            maxvalue[spe] = np.amax(self.adata[spe].obsm["normalized_log"], axis=0)
            medvalue[spe] = np.median(self.adata[spe].obsm["normalized_log"], axis=0)  

        return maxvalue, medvalue


    def _get_first_kde_valley_x(self, maxvalue):
        """Estimate first local minimum of KDE for max expression levels per species."""
        first_min_x = {}
        for spe in self.species:
            data = maxvalue[spe]
            kde = gaussian_kde(data)

            # Define a grid where the KDE will be evaluated
            x_grid = np.linspace(min(data) - 1, max(data) + 1, 10000)
            kde_values = kde(x_grid)

            # Find local minima of the KDE function
            minima = argrelextrema(kde_values, np.less)[0]

            # Assuming you are looking for the first minimum
            if len(minima) > 0:  # Ensure there is at least one minimum
                first_min_index = minima[0]  # Get the index of the first minimum
                # Get the x value of the first minimum
                first_min_x[spe] = x_grid[
                    first_min_index
                ]  
            else:
                first_min_x[spe] = None
                print("No minima found.")

        return first_min_x
    

    def _set_masking_thresholds_for_each_species(self, first_min_x):
        """Set masking thresholds for each species."""
        threshold_spe = {}
        if self.threshold_option == "manual":
            for spe in self.species:
                threshold_spe[spe] = self.threshold_surer
        elif self.threshold_option == "auto":
            for spe in self.species:
                threshold_spe[spe] = self.threshold
                if first_min_x[spe] is None:
                    print(f"first_min_x[{spe}] is None")
                else:
                    if abs(threshold_spe[spe] - first_min_x[spe]) < self.threshold_tol:
                        threshold_spe[spe] = first_min_x[spe]
                    else:
                        print(abs(threshold_spe[spe] - first_min_x[spe]), "is out of tolerance")

        return threshold_spe
    

    def _create_elementwise_threshold_mask(self, threshold_spe):
        """Create boolean mask for genes below species-specific thresholds."""
        allmask = {}

        for spe in self.species:
            # Create a Boolean mask by comparing each element of the matrix to the threshold
            allmask[spe] = self.adata[spe].obsm["normalized_log_select"] < threshold_spe[spe]

        return allmask


    def _create_max_expression_threshold_mask(self, threshold_spe):
        """Create boolean mask for genes whose max expression is below threshold."""
        maxmask = {}

        for spe in self.species:
            # Create a boolean vector (length = number of genes) indicating whether 
            # the maximum value in each column (gene) exceeds the threshold
            maxmask[spe] = (
                np.max(self.adata[spe].obsm["normalized_log_select"], axis=0) < threshold_spe[spe]
            )

        return maxmask    


    def _print_adata_shape(self, preprocessing_state):
        """Print AnnData shapes before or after preprocessing."""
        if preprocessing_state == "before":
            print("# Shape of gene expression data before masking")
            for spe in self.species:
                print(
                    spe,
                    self.adata[spe].obsm["normalized_log_select"].shape,
                )            
        else:
            print("# Shape of adata after removing columns (gene) where even the maximum value is below the threshold")
            for spe in self.species:
                print(
                    spe,
                    self.adata[spe].obsm["normalized_log_select_preprocessed_masked"].shape
                )


    def _filter_genes_by_max_expression_threshold(self, maxmask):
        """Filter genes whose max expression exceeds the threshold."""
        hvlabel_ = {}
        normalized_log_select_preprocessed_masked_data = {}

        for spe in self.species:
            mask = np.logical_not(maxmask[spe])

            # Get list of genes for which the maximum expression value is greater than a threshold value
            hvlabel_[spe] = pd.Index(self.sorted_selected_vars[spe][mask])

            # Delete columns (gene) where even the maximum value is below the threshold
            normalized_log_select_preprocessed_masked_data[spe] = self.adata[spe].obsm[
                "normalized_log_select_preprocessed"
            ][:, mask]

        return hvlabel_, normalized_log_select_preprocessed_masked_data

        
    def _filter_to_commmon_genes_across_species(self, hvlabel_):
        """Filter to genes shared across all species."""
        hvlabel = {}
        shared_genes_expression_data = {}

        # Extract genes that all species have in common out of _hvlabel_
        shared_genes = _calculate_shared_genes(self.species, hvlabel_)

        for spe in self.species:
            # Update hvlabel with shared genes
            hvlabel[spe] = pd.Index([gene for gene in hvlabel_[spe] if gene in shared_genes])
            print(
                spe, 
                "Number of shared genes",
                len(hvlabel[spe])
                )

            # Update adata object with shared genes
            mask = np.isin(hvlabel_[spe], shared_genes)
            shared_genes_expression_data[spe] = self.adata[spe].obsm[
                "normalized_log_select_preprocessed_masked"
            ][:, mask]

            print(
                spe, 
                "Shape of expression data with only shared genes left",
                shared_genes_expression_data[spe].shape
                )
                
        return hvlabel, shared_genes_expression_data


    def preprocessing(self):
        """Mask and preprocess expression data across species."""
        # Obtain maximum and median expression levels for each gene
        maxvalue, medvalue = self._obtain_maximum_and_median_gene_expression_level()

        # Estimate the distribution from the maximum expression level data for each gene using KDE 
        # And get the X-coordinate of the first local minimum of the estimated probability density function
        first_min_x = self._get_first_kde_valley_x(maxvalue)

        # Set masking thresholds for each species
        threshold_spe = self._set_masking_thresholds_for_each_species(first_min_x)

        # Generate two Boolean masks of gene expression data based on the thresholds
        allmask = self._create_elementwise_threshold_mask(threshold_spe)
        maxmask = self._create_max_expression_threshold_mask(threshold_spe)

        # Shape of gene expression data before masking
        # self._print_adata_shape(preprocessing_state="before")      

        # Masking
        for spe in self.species:
            if self.mask_option == "time_series_data":
                # Replace matrix elements below the threshold with zeros
                self.adata[spe].obsm["normalized_log_select_preprocessed"] = np.where(
                    allmask[spe], 0.0, self.adata[spe].obsm["normalized_log_select"]
                )
            elif self.mask_option == "one_time_point_data":
                # Replace gene values with 0.0 if their max expression is below threshold; 
                # otherwise keep original values
                self.adata[spe].obsm["normalized_log_select_preprocessed"] = np.where(
                    maxmask[spe][None, :], 0.0, self.adata[spe].obsm["normalized_log_select"]
                )

        # Obtain list of genes for which the maximum expression value is greater than a threshold value
        # and remove these genes from data
        hvlabel_, normalized_log_select_preprocessed_masked_data = self._filter_genes_by_max_expression_threshold(maxmask)
        for spe in self.species:
            self.adata[spe].obsm["normalized_log_select_preprocessed_masked"] = normalized_log_select_preprocessed_masked_data[spe]

        # Shape of gene expression data after preprocessing
        # self._print_adata_shape(preprocessing_state="after")

        # Create a list of genes remaining after preprocessing
        if self.gene_option == "intersection":
            # Calculate shared genes
            hvlabel, shared_genes_expression_data = self._filter_to_commmon_genes_across_species(hvlabel_)
            self.hvlabel = hvlabel
            for spe in self.species:
                # Update adata objects with shared genes
                self.adata[spe].obsm["normalized_log_select_preprocessed_masked"] = shared_genes_expression_data[spe]
        else:
            self.hvlabel = hvlabel_
        
        # Convert to Pandas DataFrame
        self.plot_normalized_log_select_preprocessed_masked = {}
        for spe in self.species:
            self.plot_normalized_log_select_preprocessed_masked[spe] = pd.DataFrame(
                self.adata[spe].obsm["normalized_log_select_preprocessed_masked"],
                index=self.adata[spe].obs.index,
                columns=self.hvlabel[spe],
            )

        return self
    

    def calculate_gene_distance_matrix(self):
        """Compute per-species gene–gene distance matrices."""
        if self.metric_option == "euclidean":
            dis_mat_ori = {}
            self.dis_mat = {}
            for spe in self.species:
                mat = (
                    self.adata[spe].obsm["normalized_log_select_preprocessed_masked"].T
                )  
                dis_mat_ori[spe] = _weighted_cdist(mat)
                self.dis_mat[spe] = dis_mat_ori[spe] / np.mean(
                    dis_mat_ori[spe]
                )  

        return self
    

    def gromov_wasserstein_ot(self):
        """Run Gromov–Wasserstein Optimal Transport (GWOT) computations."""
        # Initialize geometries
        geometries = {
            spe: _compute_geometries([jnp.array(self.dis_mat[spe])])[0]
            for spe in self.species
        }

        if self.gwot_option == "min":
            eps_high = self.high_epsilon
            converged, coupling_matrices, gw_dis_mat = _check_convergence(
                eps_high, geometries, self.iterations
            )
            _print_convergence_results(eps_high, converged)

            if converged:
                # Binary search for minimal convergent epsilon
                eps_low = self.low_epsilon
                while eps_high - eps_low > self.threshold_eps:
                    eps_mid = (eps_high + eps_low) / 2.0
                    temp_conv, _, _ = _check_convergence(
                        eps_mid, geometries, self.iterations
                    )
                    _print_convergence_results(eps_mid, temp_conv)
                    if temp_conv:
                        eps_high = eps_mid
                    else:
                        eps_low = eps_mid

                self.low_epsilon, self.high_epsilon = eps_low, eps_high
                self.best_epsilon = eps_high
                final_conv, self.coupling_matrices, self.gw_dis_mat = _check_convergence(
                    self.best_epsilon, geometries, self.iterations
                )
                assert final_conv, "Expected convergence at best_epsilon"
                print(f"Best epsilon for convergence: {self.best_epsilon} (threshold = {self.threshold_eps}).")
            else:
                print("Adjust your epsilon range.")
                self.best_epsilon = None
                print("Convergence not achieved within the given epsilon range.")

        elif self.gwot_option == "fixed":
            eps_high = self.high_epsilon
            converged, self.coupling_matrices, self.gw_dis_mat = _check_convergence(
                eps_high, geometries, self.iterations
            )
            if converged:
                self.best_epsilon = eps_high
                _print_convergence_results(self.best_epsilon, converged)
            else:
                print("GWOT solver did not converge!")

        # Convert coupling matrices to DataFrames
        self.df_coupling_matrices = {}
        for key in self.species_pairs:
            key1, key2 = key.split("_")
            self.df_coupling_matrices[key] = pd.DataFrame(
                self.coupling_matrices[key],
                index=self.hvlabel[key1],
                columns=self.hvlabel[key2],
                dtype=np.float32,
            )

        return self


    def _calculate_epsilon_entropy(self):
        """Compute epsilon entropy."""
        entropy_term = {}
        for key in self.species_pairs:
            entropy_term[key] = _calculate_entropy(self.coupling_matrices[key])

        entropy_mat = pd.DataFrame(np.nan, index=self.species, columns=self.species)

        # Populate the DataFrame with entropy values
        for key in self.species_pairs:
            spe1, spe2 = key.split("_")
            entropy_mat.loc[spe1, spe2] = entropy_term[key]  

        hyp_entropy_mat = self.best_epsilon * entropy_mat

        return hyp_entropy_mat
    

    def _calculate_gw_cost(self, hyp_entropy_mat):
        """Add epsilon entropy to regularized GW cost to get GW cost."""
        entropy_jax_array = jnp.array(hyp_entropy_mat.values)
        gw_cost_mat = self.gw_dis_mat + entropy_jax_array

        # Convert the resulting JAX array back to a pandas DataFrame
        gw_cost_df = pd.DataFrame(gw_cost_mat, index=self.species, columns=self.species)

        return gw_cost_df
    

    def _visualize_gw_cost(self, gw_cost_df):
        """Visualize GW cost matrix as a heatmap."""
        fig, ax = plt.subplots()
        sns.heatmap(gw_cost_df, ax=ax, annot=True, fmt=".3f")
        ax.set_xticklabels(self.species, rotation=0)
        ax.set_yticklabels(self.species, rotation=0)
        ax.set_title("GW cost")
        plt.show()


    def _calculate_entropy_gw_distance(self):
        """Take the square root of GW cost to get entropy GW distance."""
        hyp_entropy_mat = self._calculate_epsilon_entropy()

        gw_cost_df = self._calculate_gw_cost(hyp_entropy_mat)

        entropy_gw_distance = np.sqrt(gw_cost_df)

        return entropy_gw_distance
    

    def _visualize_entropy_gw_distance(self, entropy_gw_distance):
        """Visualize entropy-regularized GW distance as a heatmap."""
        ax = sns.heatmap(entropy_gw_distance, annot=True, fmt=".3f")
        if self.data_option == "dataset2":
            ax.set_xticklabels(self.species_labels, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(self.species_labels, rotation=0)
        ax.set_yticklabels(self.species_labels, rotation=0)
        plt.title("Entropy GW distance")
        plt.show()


    def _calculate_sinkhorn_entropy_gw_distance(self, entropy_gw_distance):
        """Compute Sinkhorn entropy GW distance (transcriptomic discrepancy array)."""
        df = entropy_gw_distance

        # Create a matrix of diagonal values repeated
        diag = np.diag(df)

        # Use broadcasting to compute the required transformation
        result = df - 0.5 * (diag[:, np.newaxis] + diag[np.newaxis, :])  # Transcriptomic distance array

        # Symmetrization
        sym_result = (result + result.T) / 2.0  # Transcriptomic discrepancy array

        # Convert to DataFrame
        sinkhorn_entropy_gw_distance = pd.DataFrame(sym_result, columns=df.columns, index=df.index)

        return sinkhorn_entropy_gw_distance


    def _visualize_sinkhorn_entropy_gw_distance(self, sinkhorn_entropy_gw_distance):
        """Visualize Sinkhorn entropy GW distance (transcriptomic discrepancy array)."""
        ax = sns.heatmap(sinkhorn_entropy_gw_distance, annot=True, fmt=".3f")
        if self.data_option == "dataset2":
            ax.set_xticklabels(self.species_labels, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(self.species_labels, rotation=0)
        ax.set_yticklabels(self.species_labels, rotation=0)
        plt.title("Transcriptomic discrepancy array")
        plt.show()


    def _set_linkage_and_labels_for_transcriptomic_discrepancy_tree(self, sinkhorn_entropy_gw_distance):
        """Compute linkage matrix and labels for the transcriptomic discrepancy tree."""
        labels = self.species_labels
        condensed = squareform(sinkhorn_entropy_gw_distance)
        linked_sinkhorn = linkage(condensed, "single")

        # X = np.array(
        #     [
        #         [0.0],
        #         [1.0],
        #         [2.0],
        #         [3.0],
        #         [4.0],
        #         [5.0],
        #     ]
        # )
        X = np.arange(len(self.species)).reshape(-1, 1)

        ordered_linkage = optimal_leaf_ordering(linked_sinkhorn, pdist(X))

        return ordered_linkage, labels


    def _generate_transcriptomic_discrepancy_tree(self, ordered_linkage, labels):
        """Visualize the transcriptomic discrepancy tree (hierarchical clustering dendrogram)."""
        custom_color = "#000000"

        fig, ax = plt.subplots(figsize=(4.5, 6))
        dendrogram(
            ordered_linkage,
            orientation="left",
            labels=labels,
            show_leaf_counts=True,
            color_threshold=np.inf,  # Color all links uniformly
            link_color_func=lambda _: custom_color,
        )

        # Aesthetic adjustments
        ax.set_title("Transcriptomic discrepancy tree")
        ax.set_xlabel("Transcriptomic discrepancy")

        # Clean up frame
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_visible(True)

        plt.tight_layout()
        plt.show()



    def plot_transcriptomic_discrepancy(self, raw_return_opt=False):
        """Compute and visualize transcriptomic discrepancy across species."""
        returnable_object_type_list = []

        # Entropy GW distance
        entropy_gw_distance = self._calculate_entropy_gw_distance()


        # Sinkhorn entropy GW distance (Transcriptomic discrepancy array)
        sinkhorn_entropy_gw_distance = self._calculate_sinkhorn_entropy_gw_distance(entropy_gw_distance)
        self._visualize_sinkhorn_entropy_gw_distance(sinkhorn_entropy_gw_distance)

        returnable_object_type_list = _append_to_returnable_object_type_list(
            raw_return_opt, returnable_object_type_list, "heatmap", "sinkhorn_entropy_gw_distance"
        )

        # Transcriptomic discrepancy tree
        ordered_linkage, labels = \
            self._set_linkage_and_labels_for_transcriptomic_discrepancy_tree(sinkhorn_entropy_gw_distance)
        self._generate_transcriptomic_discrepancy_tree(ordered_linkage, labels)

        returnable_object_type_list = _append_to_returnable_object_type_list(
            raw_return_opt, returnable_object_type_list, "dendrogram", "ordered_linkage"
        )

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print(f"The number of objects that can be returned is {len(returnable_object_type_list)}")
            return entropy_gw_distance, sinkhorn_entropy_gw_distance, ordered_linkage


    def normalize_otp(self):
        """Normalize optimal transport plans (gene-to-gene correspondences)."""
        self.df_coupling_matrices_normalized = {}
        for key in self.species_pairs:
            self.df_coupling_matrices_normalized[key] = (
                self.df_coupling_matrices[key] * self.df_coupling_matrices[key].shape[0] * 100
            )
            self.df_coupling_matrices_normalized[key].columns.name = "gene"  # temporary measures


        # For each gene, create a two-column Dataframe with the genes ordered by otp value and their otp values 
        dfp_coupling_T = {}
        self.dfp_ss = {}

        for key in self.species_pairs:
            dfp_coupling_T[key] = pl.from_pandas(
                self.df_coupling_matrices_normalized[key].T,
                include_index=True,  # Assign 'gene' as df_coupling_matrices_normalized[key].columns.name
            )
            # Initialize a nested dictionary for each key
            # ss stands for "sliced and sorted"
            self.dfp_ss[key] = {}
            for column in dfp_coupling_T[key].select(pl.exclude("gene")).columns:
                self.dfp_ss[key][column] = (
                    dfp_coupling_T[key]
                    .select(pl.col("gene"), pl.col(column))
                    .sort(column, descending=True)
                )

        return self
    

    def dashboard(self, target_species_pairs, target_genes, top_n):
        """Generate a dashboard displaying top corresponding genes between species."""
        spe1, spe2 = target_species_pairs.split('_')
        spe_label_1 = self.species[self.species.index(spe1)]
        spe_label_2 = self.species[self.species.index(spe2)]
        df_message = {}
        df_eyetest = {}
        
        for target in target_genes:
            try:
                df_message[target] = f"{spe_label_1} -> {spe_label_2}: {target}"
                df_eyetest[target] = (
                 self.dfp_ss[target_species_pairs][target]
                    .with_columns(Gene=pl.col("gene"))
                    .with_columns(
                        Value=pl.col(target)
                        .round(4)
                        .map_elements(lambda x: f"{x:.4f}", return_dtype=pl.String)
                        )
                    .select("Gene", "Value")
                    .with_row_index(offset=1)
                    .head(top_n)
                )
            except KeyError:
                print("KeyError")

        message_df_pairs = [f"{df_message[target]}\n{df_eyetest[target]}" for target in target_genes]

        try:
            _print_side_by_side(*message_df_pairs, width=36)
        except NameError:
            print("NameError")


    def _create_a_list_of_genes_to_be_plotted(self, target_species_pairs, target_genes, top_n):
        """Create a nested list of genes to be plotted."""
        gene_lists = [target_genes]

        for rank in range(top_n):
            row = []
            for gene in target_genes:
                # Get the gene corresponding to the target genes
                top_genes = self.dfp_ss[target_species_pairs][gene].select("gene").to_series().to_list()
                if rank < len(top_genes):
                    row.append(top_genes[rank])
                else:
                    row.append(None)  # If data does not exist, put None.
            gene_lists.append(row)

        return gene_lists


    def plot_corresponding_gene_expressions(
            self, target_species_pairs, spe_gene_dict, target_genes, top_n, title_fontsize=14
    ):
        """Plot cross-species gene expression correspondences in a grid layout."""
        # Split target species pairs
        spe1, spe2 = target_species_pairs.split("_")

        # Create a list of species names corresponding to each row
        species_names = _generate_list(spe1, spe2, top_n)

        # Creat a list of genes to be plotted
        gene_lists = self._create_a_list_of_genes_to_be_plotted(target_species_pairs, target_genes, top_n)

        # Make a list of what gene notations each species corresponding to each row follows
        title_formats = _generate_list(spe_gene_dict[spe1], spe_gene_dict[spe2], top_n)

        # Extract all cell names for each species
        cells = {}
        for spe in self.species:
            cells[spe] = self.adata[spe].obs.index.to_list()


        # Number of rows is determined by the number of gene lists
        rows = len(gene_lists)  
        # Number of columns is determined by the length of any gene list (assuming equal length)
        cols = len(gene_lists[0])  

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharey=True)
        axes = axes.flatten() if rows * cols > 1 else [axes]

        # Loop over each row (species)
        for row_idx, gene_list in enumerate(gene_lists):
            species_name = species_names[row_idx]  # Select species based on row index
            format_type = title_formats[row_idx]  # Select title format for the row
            for col_idx, gene_name in enumerate(gene_list):
                show_xlabel = row_idx == rows - 1  # Show xlabel only in the last row
                show_ylabel = col_idx == 0  # Show ylabel only in the first column
                if self.data_option == "dataset1":
                    _plot_gene_expression_dataset1(
                        self.species,
                        self.species_labels,
                        species_name,
                        gene_name,
                        cells,
                        self.plot_normalized_log_select_preprocessed_masked,
                        self.data_option,
                        ax=axes[row_idx * cols + col_idx],
                        title_fontsize=title_fontsize,
                        show_xlabel=show_xlabel,
                        show_ylabel=show_ylabel,
                        format_type=format_type,
                    )
                else:
                    _plot_gene_expression(
                        self.species,
                        self.species_labels,
                        species_name,
                        gene_name,
                        cells,
                        self.plot_normalized_log_select_preprocessed_masked,
                        self.data_option,
                        ax=axes[row_idx * cols + col_idx],
                        title_fontsize=title_fontsize,
                        show_xlabel=show_xlabel,
                        show_ylabel=show_ylabel,
                        format_type=format_type,
                    )

        # Hide unused subplots if necessary
        for j in range(rows * cols, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


    def _create_a_corresponding_gene_list_for_heatmap(self, target_species_pairs, target_genes, top_n):
        """Create a list of corresponding genes to visualize in a heatmap."""
        corresponding_genes = {}
        heatmap_genes = []

        for key in target_genes:
            corresponding_genes[key] = (
                self.dfp_ss[target_species_pairs][key]["gene"].head(top_n).to_list()
            )
            heatmap_genes += corresponding_genes[key]

        return heatmap_genes
    

    def _create_reference_gene_expression_dataframe(self, dataset1_bool=False):
        """
        Create reference DataFrames for gene expression comparison.
        If `dataset1_bool` is True, each DataFrame is averaged over pairs of equivalent time points (used for time-series data in dataset1). Otherwise, the original expression DataFrames are returned as-is.
        """
        reference_df = {}

        if dataset1_bool:
            # Create a new data frame averaged over two equivalent time-points
            for key in self.species:
                reference_df[key] = _merge_pairs_with_average(self.plot_normalized_log_select_preprocessed_masked[key])
        else:
            for key in self.species:
                reference_df[key] = self.plot_normalized_log_select_preprocessed_masked[key]

        return reference_df


    def _create_gene_expression_integrated_dataframe(
            self, target_species_pairs, target_genes, top_n, spe1, spe2, reference_df
    ):    
        """Create integrated DataFrames of target and corresponding gene expressions."""
        # Obtain the expression level of the target genes
        integrated_df1 = reference_df[spe1][target_genes]
        integrated_df1_transposed = integrated_df1.T

        # Obtain corresponding genes
        heatmap_genes = self._create_a_corresponding_gene_list_for_heatmap(
            target_species_pairs, target_genes, top_n
            )

        # Obtain the expression level of the corresponding genes
        integrated_df2 = reference_df[spe2][heatmap_genes]
        integrated_df2_transposed = integrated_df2.T

        return integrated_df1_transposed, integrated_df2_transposed
    

    def _plot_heatmap(self, spe, cells, spe_gene_dict, dataset1_bool, transposed_df, vmin, vmax):
        """Plot heatmap of gene expression data for a given species."""
        num_genes = transposed_df.shape[0]
        num_cells = transposed_df.shape[1]

        # parameter tuning
        if num_cells < 20:
            cell_size = 0.6 
            fig_width = num_cells * cell_size
            fig_height = num_genes * cell_size
            square_option = True
            linewidths = 0.5
            title_fontsize = cell_size * 0.5 * 72
            gene_fontsize = cell_size * 0.32 * 72  # inch to point (1 inch = 72 pt)
            cbar_fontsize = cell_size * 0.3 * 72
            pad = cell_size * 0.3
            width = cell_size * 0.4
            space = width * 1.2 / fig_width

        else:
            fig_width = max(num_cells * 0.015, 40)
            fig_height = max(3.5, min(num_genes * 1.2, 90))
            square_option = False
            linewidths = 0

            if num_genes<= 1:
                title_fontsize = 56
                gene_fontsize = 42
                cbar_fontsize = 34
                width = 1
                space = 0.03
                pad = 10
            else:
                cell_height = fig_height / num_genes
                title_fontsize = cell_height * 0.6 * 72
                gene_fontsize = cell_height * 0.4 * 72
                cbar_fontsize = cell_height * 0.36 * 72
                pad = cell_height * 0.4
                width = cell_height * 0.5
                space = width * 1.2 / fig_width


        # Automatic generation of color bar ticks
        if vmin is not None and vmax is not None:
            cbar_ticks = np.linspace(math.ceil(vmin), math.floor(vmax), num=5)
        else:
            cbar_ticks = None  # Leave it to seaborn


        # Plot start
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns_heatmap = sns.heatmap(
            transposed_df,
            ax=ax,
            cmap="coolwarm",
            annot=False,
            fmt=".1f",
            linewidths=linewidths,
            square=square_option,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )

        # y-axis label (gene name)
        formatted_genes = [
            _format_gene_name(gene, format_type=spe_gene_dict[spe]) for gene in transposed_df.index
        ]
        ax.set_yticklabels(formatted_genes, rotation=0, ha="right", fontsize=gene_fontsize)

        # x-axis label (cell name)
        cells_fontsize = gene_fontsize
        if num_cells < 20:
            if dataset1_bool:
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=cells_fontsize)
            else: 
                ax.set_xticklabels(cells[spe], fontsize=cells_fontsize)
        else:
            ax.set_xticklabels("")

        # title and y_label
        ax.set_title(f"{spe}", fontsize=title_fontsize, pad=pad)
        ax.set_ylabel("")

        # Add color bar
        cax = inset_axes(
            ax,
            width=width,
            height="100%",
            loc="upper left",
            bbox_to_anchor=(1 + space, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(sns_heatmap.collections[0], cax=cax, ticks=cbar_ticks)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=cbar_fontsize)

        plt.show()


    def _plot_heatmap_and_colorbar_separately(self, spe, cells, spe_gene_dict, dataset1_bool, transposed_df, vmin, vmax, cbar_opt):
        """Plot a gene expression heatmap without colorbar"""
        num_genes = transposed_df.shape[0]
        num_cells = transposed_df.shape[1]

        # parameter tuning
        if num_cells < 20:
            cell_size = 0.6 
            cell_height = cell_size
            fig_width = num_cells * cell_size
            fig_height = num_genes * cell_size
            square_option = True
            linewidths = 0.5
            title_fontsize = cell_size * 0.5 * 72
            gene_fontsize = cell_size * 0.32 * 72  # inch to point (1 inch = 72 pt)
            cbar_fontsize = cell_size * 0.3 * 72
            pad = cell_size * 0.3
            width = cell_size * 0.4
            space = width * 1.2 / fig_width
        else:
            fig_width = max(num_cells * 0.015, 40)
            fig_height = max(5, min(num_genes * 1.2, 90))
            square_option = False
            linewidths = 0
            cell_height = fig_height / num_genes
            title_fontsize = cell_height * 0.6 * 72
            gene_fontsize = cell_height * 0.4 * 72
            cbar_fontsize = cell_height * 0.36 * 72
            pad = cell_height * 0.4
            width = cell_height * 0.5
            space = width * 1.2 / fig_width

        # Automatic generation of color bar ticks
        if vmin is not None and vmax is not None:
            cbar_ticks = np.linspace(math.ceil(vmin), math.floor(vmax), num=5)
        else:
            cbar_ticks = None  # Leave it to seaborn


        # Plot a heatmap without a color bar
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns_heatmap = sns.heatmap(
            transposed_df,
            ax=ax,
            cmap="coolwarm",
            annot=False,
            fmt=".1f",
            linewidths=linewidths,
            square=square_option,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )

        # y-axis label (gene name)
        formatted_genes = [
            _format_gene_name(gene, format_type=spe_gene_dict[spe]) for gene in transposed_df.index
        ]
        ax.set_yticklabels(formatted_genes, rotation=0, ha="right", fontsize=gene_fontsize)

        # x-axis label (cell name)
        cells_fontsize = gene_fontsize
        if num_cells < 20:
            if dataset1_bool:
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=cells_fontsize)
            else: 
                ax.set_xticklabels(cells[spe], fontsize=cells_fontsize)
        else:
            ax.set_xticklabels("")

        # title and y_label
        ax.set_title(f"{spe}", fontsize=title_fontsize, pad=pad)
        ax.set_ylabel("")

        plt.show() 


        # Plot a color bar
        if cbar_opt:
            fig2, ax2 = plt.subplots(figsize=(cell_height, fig_height))
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
            sm.set_array([])

            cbar = fig2.colorbar(sm, cax=ax2, orientation="vertical")
            cbar.outline.set_visible(False)

            plt.show()


    def corresponding_gene_expressions_separated_heatmap(
        self,
        target_species_pairs,
        spe_gene_dict,
        target_genes,
        top_n,
        dataset1_bool=False,
        cbar_separate_opt=False,
        raw_return_opt=False,
    ):
        """Generate individual heatmaps for each target gene and its cross-species correspondences."""
        # Split species pair
        spe1, spe2 = target_species_pairs.split("_")

        # Obtain cell names
        cells = {spe: self.adata[spe].obs.index.to_list() for spe in self.species}

        # Create reference gene expression DataFrames
        reference_df = self._create_reference_gene_expression_dataframe(dataset1_bool)

        # Obtain the integrated expression matrices (for vmin/vmax unification)
        integrated_df1_transposed, integrated_df2_transposed = self._create_gene_expression_integrated_dataframe(
            target_species_pairs, target_genes, top_n, spe1, spe2, reference_df
        )

        # Unified color scale
        vmin = min(integrated_df1_transposed.min().min(), integrated_df2_transposed.min().min())
        vmax = max(integrated_df1_transposed.max().max(), integrated_df2_transposed.max().max())

        # Container for raw-return option
        returned_dfs = []

        # Plot a heatmap for each target gene
        for idx, target_gene in enumerate(target_genes):
            # 1. Expression in species 1
            df1 = reference_df[spe1][target_gene]
            df1_transposed = pd.DataFrame(df1).T
            df1_transposed.index.name = "gene"

            # 2. Corresponding genes in species 2
            heatmap_genes = self.dfp_ss[target_species_pairs][target_gene]["gene"].head(top_n).to_list()
            df2 = reference_df[spe2][heatmap_genes]
            df2_transposed = df2.T

            if cbar_separate_opt:
                # Show colorbar only for the final plot
                cbar_opt = (idx == len(target_genes) - 1)
                self._plot_heatmap_and_colorbar_separately(
                    spe1, cells, spe_gene_dict, dataset1_bool, df1_transposed, vmin, vmax, cbar_opt=False
                )
                self._plot_heatmap_and_colorbar_separately(
                    spe2, cells, spe_gene_dict, dataset1_bool, df2_transposed, vmin, vmax, cbar_opt=cbar_opt
                )
            else:
                self._plot_heatmap(spe1, cells, spe_gene_dict, dataset1_bool, df1_transposed, vmin, vmax)
                self._plot_heatmap(spe2, cells, spe_gene_dict, dataset1_bool, df2_transposed, vmin, vmax)

            if raw_return_opt:
                returned_dfs.append((df1_transposed, df2_transposed))

        # Return raw data if requested
        if raw_return_opt:
            print("# Raw data summary")
            print(f"The number of target genes is {len(target_genes)}")
            print("Each pair (df1_transposed, df2_transposed) corresponds to one target gene.")
            print("Both elements are pd.DataFrame instances.")
            return returned_dfs

