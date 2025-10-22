import csv
import functools
import math
import os
from typing import Literal
import statistics
import logging
# logging.getLogger('absl').setLevel(logging.ERROR)
os.environ['ABSL_FLAGS'] = '--stderrthreshold=error'

import anndata
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import seaborn as sns
import sklearn.decomposition
import sklearn.manifold
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ott.geometry import geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde
import screcode


def configure_platform(platform="metal"):
    """configure_platform

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
    """_calculate_intersections
    
    Function to create a list of genes common to all Species among the transcription_factors

    This is used in Data class function "read_tf"
    """    
    intersections = []
    for spe in species_list:
        intersect = np.intersect1d(transcription_factors, adata[spe].var.index)
        print(spe, intersect.shape)
        intersections.append(intersect)
    return functools.reduce(np.intersect1d, intersections)


def _generate_list(A, B, n):
    """_generate_list

    This is used in SpeciesOT class function "plot_corresponding_gene_expressions"
    """
    return [A] + [B] * n


def _format_gene_name(gene_name, format_type):
    """_format_gene_name
    
    Function to format the gene name according to the given format type
    
    This is used in SpeciesOT class function "plot_corresponding_gene_expressions", 
    "corresponding_gene_expressions_heatmap", and "plot_pca_highlighting_genes_retained_after_preprocessing"
    """
    if format_type == "all_capital_italic":
        return f"$\\it{{{gene_name.upper()}}}$"
    elif format_type == "capitalized_italic":
        return f"$\\it{{{gene_name.capitalize()}}}$"
    return gene_name  # Default to unformatted


def _format_gene_name_for_plotly(gene_name, format_type):
    """_format_gene_name_for_plotly
    
    Function to format the gene name according to the given format type
    
    This is used in the function "_get_display_gene_name"
    """
    if format_type == "all_capital_italic":
        return f"<i>{gene_name.upper()}</i>"
    elif format_type == "capitalized_italic":
        return f"<i>{gene_name.capitalize()}</i>"
    return gene_name  # Default to unformatted


def _get_display_gene_name(gene, spe, spe_gene_dict, plotly_opt=False):
    """_get_display_gene_name
    
    Function to apply common name conversion and formatting for all genes

    This is used in SpeciesOT class function "plot_pca_highlighting_genes_retained_after_preprocessing_plotly"
    """
    if gene == "PRDM1":
        gene_display = "BLIMP1"
    elif gene == "TBXT":
        gene_display = "T"
    else:
        gene_display = gene

    if plotly_opt:
        return _format_gene_name_for_plotly(gene_display, spe_gene_dict[spe])
    else:
        return _format_gene_name(gene_display, spe_gene_dict[spe])


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
    """_plot_gene_expression
    
    Function to plot gene expression data for a given species and gene on a specified axis
    
    This is used in SpeciesOT class function "plot_corresponding_gene_expressions"
    """
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
    if data_option == "dataset1":
        ylabel = "log2(RPM+1)"
        xlabel = "Sampling timing"
        ax.set_ylim(0, 14)
    elif data_option == "dataset2":
        ylabel = "log2(RP100K+1)"
        xlabel = "Single-cells"
        ax.set_ylim(0, 8)

    if show_ylabel:
        ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(xlabel)


# Plot dataset1 data fully adjusted to time series data
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
    """_plot_gene_expression_dataset1
    
    Function to plot dataset1 gene expression data fully adjusted to time series data for a given species and gene on a specified axis
    
    This is used in SpeciesOT class function "plot_corresponding_gene_expressions"
    """
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
    """assign_colors

    Assign colors to labels based on target labels.
    """
    return [
        "blue" if label in target_labels else default_color
        for label in labels
    ]


def _calculate_shared_genes(species_list, hvlabel_):
    """_calculate_intersections
    
    Function to create a list of genes common to all Species among hvlavel_

    This is used in SpeciesOT class function "_filter_to_commmon_genes_across_species"
    """    
    intersections = []
    for spe in species_list:
        masked_genes = hvlabel_[spe]
        intersections.append(masked_genes)
    shared_genes = functools.reduce(np.intersect1d, intersections)
    return shared_genes


def _weighted_cdist(X, w=None):
    """weighted_cdist
    
    This is used in SpeciesOT class function "calculate_dismat"
    """
    # Ensure inputs are numpy arrays for efficient computation
    X = np.asarray(X)

    # If no weights are provided, use default weights of 1 for each dimension
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


def _generate_weights(full_size, missing_indices, high_value, low_value):
    """_generate_weights
    
    Function to generate weights

    This is used in SpeciesOT class function "calculate_dismat"
    """
    # Initialize all weights with the low value
    weights = np.full(full_size, low_value)

    # Set the high value for the first few components based on the structure
    # Assuming the first four components should have the high value in the full vector
    high_value_count = 4

    # Update the first components to the high value
    weights[:high_value_count] = high_value

    # Adjust for missing components
    # Remove the weights for the missing components, starting from the end to keep the indices correct
    for index in sorted(missing_indices, reverse=True):
        if index < full_size:  # Ensure the missing index is within the range
            weights = np.delete(weights, index)
        # Adjust high_value_count if the missing index is within the high value range
        if index < high_value_count:
            high_value_count -= 1
        
    # Ensure the remaining weights beyond high_value_count are set to low_value
    weights[high_value_count:] = low_value

    return weights


def _compute_geometries(distance_matrices):
    """_compute_geometries
    
    Function to create a list of Geometry instances with each matrix of distance_matrices as cost_matrix

    This is used in SpeciesOT class function "gromov_wasserstein_ot"
    """
    return [geometry.Geometry(cost_matrix=dm) for dm in distance_matrices]


def _print_convergence_results(epsilon, converged):
    """_print_convergence_results

    Function to print convergence results

    This is used in SpeciesOT class function "gromov_wasserstein_ot"
    """
    status = "converged" if converged else "non-converged"
    print(f"epsilon = {epsilon:.7f} {status}")
    # Additional details can be printed here as needed


def _check_convergence(epsilon, geometries, iterations):
    """_check_convergence

    Function to check convergence
    
    This is used in SpeciesOT class function "gromov_wasserstein_ot"
    """
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
            sinkhorn_converged = bool(
                result.linear_convergence[n_converged_iterations - 1]
            )
            gw_converged = result.converged

            pair_converged = sinkhorn_converged and gw_converged
            all_converged &= pair_converged

            species_pair_key = f"{spec_i}_{spec_j}"
            coupling_matrices[species_pair_key] = result.matrix
            gw_dis_mat = gw_dis_mat.at[idx_i, idx_j].set(result.reg_gw_cost)
            
    return all_converged, coupling_matrices, gw_dis_mat


def _calculate_entropy(p):
    """_calculate_entropy
    
    Function to calculate entropy

    This is used in SpeciesOT class function "dendrogram", "plot_transcriptomic_discrepancy", "outputs_for_paper"
    """
    # Use np.where to handle log(0) by setting the result to 0 where P_ij is 0
    log_p = np.where(p > 0, np.log(p + 1e-12), 0)
    h_p = -np.sum(p * (log_p - 1))
    return h_p


def _append_to_returnable_object_type_list(raw_return_opt, returnable_object_type_list, object_type, object_name):
    """_append_to_returnable_object_type_list
    
    Function to append object type to retunable_object_list

    This is used in SpeciesOT function "plot_transcriptomic_discrepancy" and "outputs_for_paper"
    """
    if raw_return_opt:
        object_discript_dict = {
            "heatmap": "pd.DataFrame",
            "dendrogram": "a linkage matrix (np.darray)",
            "jnp_heatmap": "jnp.darray"
        }
            
        returnable_object_type_list.append(object_discript_dict[object_type])
        print(f"The returned object (`{object_name}`) is {returnable_object_type_list[-1]}")

    return returnable_object_type_list


def csv2list(path):
    """csv2list
    
    Function to generate a gene list from a single row of genes on a csv file

    Args:
        - path (str) : the path of csv file

    Returns:
        gene_list (list)

    Examples:
        >>> gene_list = csv2list("dashboard_genelist.csv)
    """
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gene_list = row
    return gene_list


def _print_side_by_side(*args, width=20):
    """_print_side_by_side
    
    Function to create a dashboard from data

    This is used in SpeciesOT class function "dashboard"
    """    
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


def _merge_pairs_with_average(df):
    """_merge_pairs_with_average
    
    Function to create a new data frame by taking the average of two rows of data each.

    This is used in SpeciesOT class function "corresponding_gene_expressions_heatmap"
    """    
    new_rows = []
    new_index = []

    # Take the average of two rows each. (2k, 2k+1)
    for i in range(0, len(df), 2):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        avg_row = (row1 + row2) / 2

        # Remove trailing “#1”, “#2”, “1”, “2” from labels
        original_label = df.index[i]

        if original_label.endswith("#1") or original_label.endswith("#2"):
            label = original_label[:-2]  
        elif original_label.endswith("1") or original_label.endswith("2"):
            label = original_label[:-1]  
        else:
            label = original_label  

        new_rows.append(avg_row)
        new_index.append(label)

    # Creat a new dataframe
    merged_df = pd.DataFrame(new_rows, index=new_index)

    return merged_df


class Config:
    """Config

    Class for storing parameters
    """
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

        self.species_pairs = species_pairs
        for spe1 in species:
            for spe2 in species:
                self.species_pairs.append(f"{spe1}_{spe2}")




class Data(Config):
    """Data
    
    Class for retrieving data.
    """
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
        """_set_index_and_columns_names_to_none
        
        Function to set the names of index and columns to none

        This is used in Data class function "read_csv" 
        """
        df.index.name = None
        df.columns.name = None
        return df


    def _creat_adata(self, df_csv, spe):
        """_creat_adata
        
        Function to create adata sorted alphabetically for genes with non-zero expression

        This is used in Data class function "read_csv" 
        """
        sorted_genes = {}
        adata = {}
        gene_expression_sums = {}

        adata[spe] = anndata.AnnData(df_csv[spe])  
        adata[spe].obs_names = df_csv[spe].index
        adata[spe].var_names = df_csv[spe].columns
        sorted_genes[spe] = adata[spe].var.index.sort_values()
        adata[spe] = adata[spe][:, sorted_genes[spe]]
        gene_expression_sums[spe] = np.array(adata[spe].X.sum(axis=0)).flatten()
        adata[spe] = adata[spe][:, gene_expression_sums[spe] > 0]  
        # print(spe, adata[spe].shape)

        return adata[spe]
    


    def read_csv(self):
        """read_csv
        
        Read csv files

        Args:
            - input_dir (path): the path of input directory
            Default to "../data/".

        Returns:
            self

        Example:
            >>> data = data.read_csv()
        """
        if self.data_option == "dataset1" or self.data_option == "dataset2":
            INPUT_DIR = "../data"
        elif self.data_option == "custom":
            INPUT_DIR = "../custom"

        df_csv = {}
        self.adata = {}

        if self.data_option == "dataset1":
            for spe in self.species:
                df_csv[spe] = pd.read_csv(f"{INPUT_DIR}/dataset1_{spe}.csv", header="infer", index_col="gene")
                if spe == "mouse" and self.mouse_option == "drop":
                    df_csv[spe] = df_csv[spe].drop(["mESC#1", "mESC#2"], axis=1)
                df_csv[spe] = self._set_index_and_columns_names_to_none(df_csv[spe])
                self.adata[spe] = self._creat_adata(df_csv, spe)
                df_csv[spe] = df_csv[spe].T
                df_csv[spe] = self._set_index_and_columns_names_to_none(df_csv[spe])
                df_csv[spe] = df_csv[spe].astype(np.float32)
                self.adata[spe] = self._creat_adata(df_csv, spe)
        
        elif self.data_option == "dataset2":
            for spe in self.species:
                df_csv[spe] = pd.read_csv(f"{INPUT_DIR}/dataset2_{spe}.csv", header="infer", index_col="CellID")  
                df_csv[spe] = self._set_index_and_columns_names_to_none(df_csv[spe])
                self.adata[spe] = self._creat_adata(df_csv, spe)
                if spe == "mouse":
                    self.adata[spe] = self.adata[spe].copy()
                    self.adata[spe].var.index = self.adata[spe].var.index.str.upper()

        elif self.data_option == "custom":
            for spe in self.species:
                df_csv[spe] = pd.read_csv(f"{INPUT_DIR}/{spe}.csv", header="infer", index_col="CellID")
                df_csv[spe] = self._set_index_and_columns_names_to_none(df_csv[spe])
                self.adata[spe] = self._creat_adata(df_csv, spe)

        else:
            print(self.data_option)


        for spe in self.species:
            self.adata[spe].var.index.name = "gene"


        return self



    def normalization(self):
        """normalization

        Function to reduce noise and normalize data

        Returns:
            - self

        Outputs:
            - Shape of adata after noise reduction and normalization

        Examples:
            >>> data = data.normalization()
        """
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



    def check_gene_exp(self, species, gene, threshold=None):
        """check_gene_exp
        
        Function to check gene expression levels for a given species and genes

        Args:
            - species (str): The species labels ("human” etc.)
            - gene(str): The gene name to check gene expression levels for.
            - threshold(float, optimal): If provided, a horizontal line at y=threshold 
            will be added to the plot. Defaults to None.
 
        Examples:
            >>> data.gene_exp_check()
        """
        self.plot_normalized_log= {}

        for spe in self.species:
            self.plot_normalized_log[spe] = pd.DataFrame(
                self.adata[spe].obsm["normalized_log"],
                index=self.adata[spe].obs.index,
                columns=self.adata[spe].var.index,
            )

        try:
            fig, ax = plt.subplots()
            self.plot_normalized_log[species][gene].plot(ax=ax, marker="o")
            ax.set_title(f"{species} {gene}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

            if threshold is not None:
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'threshold={threshold}')
                ax.legend()

        except KeyError:
            print(f"Warning: Column '{gene}' not found in the Dataframe for species `{species}`")
        

        plt.show()



    def plot_read(self):
        """plot_read
        
        Function to check gene expression data

        Outputs:
            - Shape of Polars DataFrame with cell names added to the beginning of gene expression data
            - Shape of gene expression Pandas DataFrame
            - Simplified version of the expression matrix

        Examples:
            >>> data.plot_read()
        """
        # Create a Polars DataFrame with only cell names
        arr = {}
        dfp = {}
        for spe in self.species:
            arr[spe] = np.array(self.adata[spe].obs.index.tolist())
            dfp[spe] = pl.from_numpy(arr[spe], schema=["cells"])


        # Gene expression data + cell name Polars DataFrame
        plot_df0 = {}
        for spe in self.species:
            plot_df0[spe] = pl.from_numpy(
                self.adata[spe].obsm["normalized_log"],
                schema=self.adata[spe].var.index.tolist(),
                orient="row",
            ).with_columns(dfp[spe])


        # plot_df0 with cells column moved to the top
        print("# Shape of Polars DataFrame with cell names added to the beginning of gene expression data")
        plot_dfp = {}
        for spe in self.species:
            plot_dfp[spe] = plot_df0[spe].select(pl.col("cells"), pl.exclude("cells"))
            print(spe, plot_dfp[spe].shape)


        # Gene expression data for Pandas version
        plot_df0_ = {}
        for spe in self.species:
            plot_df0_[spe] = pd.DataFrame(
                self.adata[spe].obsm["normalized_log"],
                index=self.adata[spe].obs.index,
                columns=self.adata[spe].var.index,
            )
            print(
                spe, 
                "Shape of the gene expression matrix",
                plot_df0_[spe].shape
                )
            print(f"# Showing the first 5 rows (cells) and first 6 columns (genes) of the expression matrix for '{spe}'")
            print(plot_df0_[spe].iloc[:5, :6])



    def _get_sorted_selected_gene_expression(self, gene_selection, spe):
        """_get_sorted_selected_gene_expression
        
        Function to obrain transcription factors from inputted file

        This is used in Data class function "read_tf"
        """
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
        """read_tf
        
        Read human transcription factors genes from hTF.txt in TFs directory

        Returns:
            - self

        Outputs:
            - gene_selection.shape

        Example:
            >>> data = data.read_tf()
        """
        df_TF = pd.read_csv("../TFs/hTF.txt", delimiter="\t", index_col="gene_id")
        df_TF.sort_values("gene_type", inplace=True)
        transcription_factors = df_TF["gene_type"].values

        # Select genes based on the gene_option
        if self.gene_option == "intersection":
            # Calculate intersections of transcription factors with species genes
            gene_selection = _calculate_intersections(self.species, transcription_factors, self.adata)
        elif self.gene_option == "distinct":
            gene_selection = transcription_factors

        print("#genes in hTF.txt")
        print(gene_selection.shape[0])

        self.sorted_selected_vars = {}
        for spe in self.species:
            (
                self.adata[spe].obsm["normalized_log_select"], 
                self.sorted_selected_vars[spe] 
            ) = self._get_sorted_selected_gene_expression(gene_selection, spe)

        return self



class SpeciesOT(Data):
    """SpeciesOT

    Cross Species Optimal Transport
    """
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
        """_generate_gene_exprerssion_histogram
        
        Function to obtain maximum and median expression levels for each gene

        This is used in SpeciesOT class function "visualization" and "preprocessing"
        """ 
        maxvalue = {}
        medvalue = {}
        for spe in self.species:
            maxvalue[spe] = np.amax(self.adata[spe].obsm["normalized_log"], axis=0)
            medvalue[spe] = np.median(self.adata[spe].obsm["normalized_log"], axis=0)  

        return maxvalue, medvalue



    def _generate_gene_exprerssion_histogram(self, data_pattern, max_or_med, value):
        """_generate_gene_exprerssion_histogram
        
        Function to generate gene expressioni histogram

        This is used in SpeciesOT class function "visualization"
        """ 
        fig, ax = plt.subplots()

        for spe in self.species:
            sns.histplot(value[spe], ax=ax, bins=100, kde=True)        

        if max_or_med == "max":
            ax.set_title("Histogram of the maximum expression level of each gene")
            ax.set_xlabel("Maximum expression level")
        elif max_or_med == "med":
            ax.set_title("Histogram of the median expression level of each gene")
            ax.set_xlabel("Median expression level")

        ax.set_ylabel("Frequency")
        ax.legend(self.species_labels)

        if data_pattern == "dataset2":
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 1200)
        elif data_pattern == "dataset1":
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 600)
        else:
            print("--")

        plt.show()



    def _generate_gene_exprerssion_histogram_with_threshold(self, data_pattern, max_or_med, value):
        """_generate_gene_exprerssion_histogram_with_threshold
        
        Function to generate gene expression histogram with threshold line

        This is used in SpeciesOT class function "visualization"
        """ 
        fig, ax = plt.subplots()

        for spe in self.species:
            sns.histplot(value[spe], ax=ax, bins=100, kde=True)        

        if max_or_med == "max":
            ax.set_title("Histogram of the maximum expression level of each gene")
            ax.set_xlabel("Maximum expression level")
        elif max_or_med == "med":
            ax.set_title("Histogram of the median expression level of each gene")
            ax.set_xlabel("Median expression level")

        ax.set_ylabel("Frequency")
        ax.legend(self.species_labels)
        ax.axvline(x=self.threshold, color="red")

        if data_pattern == "dataset1":
            if max_or_med == "max":
                ax.text(self.threshold * 1.05, 600, "log2(RPM+1)=" + str(self.threshold))
                ax.set_ylim(0, 800)
            elif max_or_med == "med":
                ax.text(self.threshold * 1.05, 450, "log2(RPM+1)=" + str(self.threshold))
                ax.set_ylim(0, 600)

            ax.set_xlim(0, 14)

        elif data_pattern == "dataset2":
            ax.text(self.threshold * 1.05, 600, "log2(RPM+1)=" + str(self.threshold))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 800)

        else:
            print("--")

        plt.show()



    def _generate_kernel_density_estimate_plot_of_maximum_gene_expression(self, maxvalue):
        """_generate_kernel_density_estimate_plot_of_maximum_gene_expression
        
        Function to generate kernel density estimate plot of maximum expresssion of each gene

        This is used in SpeciesOT class function "visualization"
        """ 
        fig, ax = plt.subplots()

        for spe in self.species:
            data = maxvalue[spe]
            title = "kdeplot of maximum expresssion of each gene"
            xlabel = "Maximum expression level"
            kde = gaussian_kde(data)
            x_grid = np.linspace(min(data), max(data), 1000)
            density_values = kde(x_grid)
            ax.plot(x_grid, density_values, label="KDE")

        ax.set_title(title)
        ax.legend(self.species_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_xlim(0, 14)
        ax.axvline(x=self.threshold, color="red")
        ax.text(self.threshold * 1.05, 0.25, f"log2(RPM+1)={self.threshold}")
        ax.set_ylim(0, 0.3)

        plt.show()



    def _generate_kernel_density_estimate_plot_of_median_gene_expression(self, medvalue):
        """_generate_kernel_density_estimate_plot_of_median_gene_expression
        
        Function to generate kernel density estimate plot of median expresssion of each gene

        This is used in SpeciesOT class function "visualization"
        """ 
        fig, ax = plt.subplots()

        for spe in self.species:
            sns.kdeplot(x=medvalue[spe], ax=ax)
        
        ax.set_title("kdeplot of median expresssion of each gene")
        ax.set_xlabel("Median expression level")
        ax.set_ylabel("Density")
        ax.legend(self.species_labels)
        ax.axvline(x=self.threshold, color="red")
        ax.text(self.threshold * 1.05, 0.15, f"log2(RPM+1)={self.threshold}")
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 0.3)

        plt.show()



    def _generate_kernel_density_estimate_with_critical_points(self, maxvalue):
        """_generate_kernel_density_estimate_with_critical_points
        
        Function to generate kernel density estimate eith critical points

        This is used in SpeciesOT class function "visualization"
        """ 
        for spe in self.species:
            fig, ax = plt.subplots(figsize=(10, 6))

            data = maxvalue[spe]
            kde = gaussian_kde(data)

            # Step 2: Define a grid where the KDE will be evaluated
            x_grid = np.linspace(min(data) - 1, max(data) + 1, 10000)
            kde_values = kde(x_grid)

            # Step 3: Find local maxima and minima of the KDE function
            # These points are where the derivative is close to 0
            maxima = argrelextrema(kde_values, np.greater)[0]
            minima = argrelextrema(kde_values, np.less)[0]

            # The 'maxima' and 'minima' arrays hold indices of the x_grid that are local maxima and minima
            # To find the x values corresponding to these indices:
            maxima_x = x_grid[maxima]
            minima_x = x_grid[minima]

            ax.plot(x_grid, kde_values, label="KDE")
            ax.plot(maxima_x, kde_values[maxima], "ro", label="Maxima")
            ax.plot(minima_x, kde_values[minima], "bo", label="Minima")
            ax.legend()
            ax.set_title(f"KDE with Critical Points: {spe}")
            plt.show()



    def visualization(self, raw_return_opt=False):
        """visualization

        Function to plot the distribution of gene expression levels

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Outputs:
            - Histogram of the maximum expression of each gene
            - Histogram of the median expression of each gene
            - Histogram of the maximum expression of each gene with threshold line
            - Histogram of the median expression of each gene with threshold line
            - Kernel density estimate plot of maximum expression of each gene
            - Kernel density estimate plot of median expression of each gene
            - Kernel density estimate with Critical Points
        
        Examples:
            >>> spe_ot.visualization()
        """
        if (
            self.data_option == "dataset2"
        ):
            data_pattern = "dataset2"
        elif (
            self.data_option == "dataset1"
        ):
            data_pattern = "dataset1"

        # Obtain maximum and median expression levels for each gene
        maxvalue, medvalue = self._obtain_maximum_and_median_gene_expression_level()

        if self.data_option != "custom":
            # Histogram of the maximum expression of each gene
            self._generate_gene_exprerssion_histogram(data_pattern, max_or_med="max", value=maxvalue)

            # Histogram of the median expression of each gene
            self._generate_gene_exprerssion_histogram(data_pattern, max_or_med="med", value=medvalue)

            # Histogram of the maximum expression of each gene with threshold line
            self._generate_gene_exprerssion_histogram_with_threshold(data_pattern, max_or_med="max", value=maxvalue)

            # Histogram of the median expression of each gene with threshold line
            self._generate_gene_exprerssion_histogram_with_threshold(data_pattern, max_or_med="med", value=medvalue)

        # Kernel density estimate plot of maximum expression of each gene
        self._generate_kernel_density_estimate_plot_of_maximum_gene_expression(maxvalue)

        # Kernel density estimate plot of median expression of each gene
        self._generate_kernel_density_estimate_plot_of_median_gene_expression(medvalue)

        # Kernel density estimate with Critical Points
        self._generate_kernel_density_estimate_with_critical_points(maxvalue)

        # Return raw data
        if raw_return_opt:
            print("# Rawa data sumamry")
            print("The number of objects that can be returned is 2")

            print("# The first returned object (`maxvalue`) is a dict")
            print("This is maximum expression levels for each gene")
            print("The key of `maxvalue` is self.species")
            print("The value of `maxvalue` is np.ndarray")

            print("# The second returned object (`medvalue`) is a dict")
            print("This is medium expression levels for each gene")
            print("The key of `medvalue` is self.species")
            print("The value of `medvalue` is np.ndarray")

            return maxvalue, medvalue



    def _get_first_kde_valley_x(self, maxvalue):
        """_get_first_kde_valley_x
        
        Function to estimate the distribution from the maximum expression level data for each given gene using kernel density estimation 
        and returns the X coordinate of the first local minimum of the estimated probability density function

        This is used in SpeciesOT class function "preprocessing"
        """
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
        """_set_masking_thresholds_for_each_species
        
        Function to set masking threshold for each species

        This is used in SpeciesOT class function "preprocessing"
        """
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
        """_create_elementwise_threshold_mask
        
        Funtion to create a Boolean mask by comparing each element of the matrix to the threshold

        This is used in SpeciesOT class function "preprocessing"
        """
        allmask = {}

        for spe in self.species:
            # Create a Boolean mask by comparing each element of the matrix to the threshold
            allmask[spe] = self.adata[spe].obsm["normalized_log_select"] < threshold_spe[spe]

        return allmask



    def _create_max_expression_threshold_mask(self, threshold_spe):
        """_create_max_expression_threshold_mask
        
        Funtion to create a boolean vector indicating whether the maximum value in each column (gene) exceeds the threshold

        This is used in SpeciesOT class function "preprocessing"
        """
        maxmask = {}

        for spe in self.species:
            # Create a boolean vector (length = number of genes) indicating whether 
            # the maximum value in each column (gene) exceeds the threshold
            maxmask[spe] = (
                np.max(self.adata[spe].obsm["normalized_log_select"], axis=0) < threshold_spe[spe]
            )

        return maxmask    
    


    def _print_adata_shape(self, preprocessing_state):
        """_print_adata_shape

        Function to print adata shape
        
        This is used in SpeciesOT class function "preprocessing"
        """
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
        """_filter_genes_by_max_expression_threshold
        
        Function to filter genes by max expression threshold

        This is used in SpeciesOT class function "preprocessing"
        """
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
        """_filter_to_commmon_genes_across_species

        Function to generate hvlabel
        
        This is used in SpeciesOT class function "preprocessing"
        """
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
        """preprocessing

        Function to mask and preprocess adata
            
        Returns:
            - self

        Outputs:
            - Shape of gene expression data before masking
            - Number of genes after preprocessing (when gene_option==="distinct")
            - Number of shared genes (when gene_option=="intersection")
            - Shape of gene expression data with only shared genes left  (when gene_option=="intersection")

        Examples:
            >>> spe_ot.preprocessing()
        """
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
            hvlabel, shared_genes_expression_data = self._filter_to_commmon_genes_across_species(self, hvlabel_)
            self.hvlabel = hvlabel
            for spe in self.species:
                # Update adata objects with shared genes
                self.adata[spe].obsm["normalized_log_select_preprocessed_masked"] = shared_genes_expression_data[spe]
        else:
            self.hvlabel = hvlabel_
            # print("# Number of genes for which the maximum expression value is greater than a threshold value")
            # for spe in self.species:
            #     print(
            #         spe,
            #         self.hvlabel[spe].shape
            #         )
        
        # Convert to Pandas DataFrame
        self.plot_normalized_log_select_preprocessed_masked = {}
        for spe in self.species:
            self.plot_normalized_log_select_preprocessed_masked[spe] = pd.DataFrame(
                self.adata[spe].obsm["normalized_log_select_preprocessed_masked"],
                index=self.adata[spe].obs.index,
                columns=self.hvlabel[spe],
            )

        return self
    


    def _perform_pca_to_check_the_consequence_of_preprocessing(self, comparison):
        """ _perform_pca_to_check_the_consequence_of_preprocessing
        
        Function to perform pca to investigate the distribution of genes remaining after preprocessing

        This is used in SpeciesOT class function "plot_pca_highlighting_genes_retained_after_preprocessing" 
        and "plot_pca_highlighting_genes_retained_after_preprocessing_plotly"
        """
        all_principal_components = []
        explained_variance_ratio_dict = {}

        # Perform PCA
        for spe in self.species:
            if comparison == "transcription_factors":
                pca = sklearn.decomposition.PCA(n_components=2)
                principal_components = pca.fit_transform(self.adata[spe].obsm["normalized_log_select"].T)
            elif comparison == "all_gene":
                pca = sklearn.decomposition.PCA(n_components=2)
                principal_components = pca.fit_transform(self.adata[spe].X.T)
            else:
                raise ValueError("Invalid value for 'comparison'. Expected 'transcription_factors' or 'all_gene'.")
            
            all_principal_components.append(principal_components)
            explained_variance_ratio_dict[spe] = pca.explained_variance_ratio_

        return all_principal_components, explained_variance_ratio_dict
        


    def _determine_a_common_axis_range_from_all_principal_component_data(self, all_principal_components):
        """_determine_a_common_axis_range_from_all_principal_component_data
        
        Function to determine a common axis range from all principal component data

        This is used in SpeciesOT class function "plot_pca_highlighting_genes_retained_after_preprocessing" 
        and "plot_pca_highlighting_genes_retained_after_preprocessing_plotly"
        """
        # Use np.vstack to vertically concatenate the numpy arrays in the list and calculate the minimum and maximum values of all data points
        all_pc_data = np.vstack(all_principal_components)
        x_min, x_max = all_pc_data[:, 0].min(), all_pc_data[:, 0].max()
        x_range = x_max - x_min
        y_min, y_max = all_pc_data[:, 1].min(), all_pc_data[:, 1].max()
        y_range = y_max - y_min

        lower_limit_of_x_axis = x_min - x_range*0.05
        upper_limit_of_x_axis = x_max + x_range*0.05
        lower_limit_of_y_axis = y_min - y_range*0.05
        upper_limit_of_y_axis = y_max + y_range*0.05

        return lower_limit_of_x_axis, upper_limit_of_x_axis, lower_limit_of_y_axis, upper_limit_of_y_axis
    


    def _get_a_list_of_genes_before_and_after_preprocessing(self, comparison):
        """_get_a_list_of_genes_before_and_after_preprocessing

        Function to get a list of genes before and after preprocessing

        This is used in SpeciesOT class function "plot_pca_highlighting_genes_retained_after_preprocessing" 
        and "plot_pca_highlighting_genes_retained_after_preprocessing_plotly"
        """
        gene_list_before_preprocessing = {}
        gene_list_after_preprocessing = {}

        for spe in self.species:
            # Get a list of genes before preprocessing
            if comparison == "transcription_factors":
                gene_list_before_preprocessing[spe] = self.sorted_selected_vars[spe].to_list()
            elif comparison == "all_gene":
                gene_list_before_preprocessing[spe] = self.adata[spe].var.index.to_list()

            # Get a list of genes after preprocessing
            gene_list_after_preprocessing[spe] = self.hvlabel[spe].to_list()

        return gene_list_before_preprocessing, gene_list_after_preprocessing
    


    def _create_a_boolean_mask_for_genes_that_are_in_both_lists(
            self, gene_list_before_preprocessing, gene_list_after_preprocessing
        ):
        """_create_a_boolean_mask_for_genes_that_are_in_both_lists

        Function to creat a boolean mask for genes that are in both lists

        This is used in SpeciesOT class function "plot_pca_highlighting_genes_retained_after_preprocessing" 
        and "plot_pca_highlighting_genes_retained_after_preprocessing_plotly"
        """
        is_gene_retained = {}
        
        for spe in self.species:
            # Convert gene_list_after_preprocessing to a set to speed up search
            gene_set_after_preprocessing = set(gene_list_after_preprocessing[spe])

            # Create a boolean mask for genes that are in both lists
            is_gene_retained[spe] = [gene in gene_set_after_preprocessing for gene in gene_list_before_preprocessing[spe]]

        return is_gene_retained
    


    def plot_pca_highlighting_genes_retained_after_preprocessing(self, comparison, labeled_genes, spe_gene_dict):
        """plot_pca_highlighting_retained_after_preprocessing
        
        Function to plot expression of all genes, with particular emphasis on genes that remained after preprocessing 

        Args:
            - comparison (str): Compared to when all genes were targeted ("all_gene"), or compared to when only transcription factor genes were targeted ("transcription_factors")
            - labeled_genes (dict): A list of genes to be labeled on the scatter plot, grouped by species
            - spe_gene_dict (dict): Gene name notation for each species.  ex. spe_gene_dict["mouse"] = "capitalized_italic", spe_gene_dict["human"] = "all_capitalized_italic"

        Outputs:
            - figure

        Examples:
            >>> spe_ot.plot_pca_highlighting_genes_retained_after_preprocessing()
        """
        # Perform PCA
        all_principal_components, explained_variance_ratio_dict = self._perform_pca_to_check_the_consequence_of_preprocessing(comparison)

        # Determine a common axis range from all principal component data
        lower_limit_of_x_axis, upper_limit_of_x_axis, lower_limit_of_y_axis, upper_limit_of_y_axis = (
            self._determine_a_common_axis_range_from_all_principal_component_data(all_principal_components)
        )

        # Get a list of genes before and after preprocessing
        gene_list_before_preprocessing, gene_list_after_preprocessing = (
            self._get_a_list_of_genes_before_and_after_preprocessing(comparison)
        )

        # Create a boolean mask for genes that are in both lists
        is_gene_retained = self._create_a_boolean_mask_for_genes_that_are_in_both_lists(
            gene_list_before_preprocessing, gene_list_after_preprocessing
        )

        for i, spe in enumerate(self.species):
            principal_components = all_principal_components[i]
            print(f"Explain Variance Ratio : {explained_variance_ratio_dict[spe]}")

            # Plotting
            plt.figure(figsize=(12, 10))

            # Plot all points in gray
            plt.scatter(principal_components[:, 0], principal_components[:, 1], color='gray', label='All genes')

            # Plot selected genes in red
            plt.scatter(principal_components[is_gene_retained[spe], 0], principal_components[is_gene_retained[spe], 1], color='red', label='retained genes')

            # Set a common axis range
            plt.xlim(lower_limit_of_x_axis, upper_limit_of_x_axis)
            plt.ylim(lower_limit_of_y_axis, upper_limit_of_y_axis)

            # Add label
            for gene in labeled_genes[spe]:
                try:
                    # Confirm that the gene name exists in gene_list_before_preprocessing and obtain its index
                    gene_index = gene_list_before_preprocessing[spe].index(gene)

                    # Obtain the coordinates on PCA corresponding to that gene
                    x = principal_components[gene_index, 0]
                    y = principal_components[gene_index, 1]

                    # Plot specific genes with blue star marks
                    plt.scatter(x, y, color='blue', marker='*', s=200, edgecolor='black', zorder=5)

                    # Apply common name conversion and formatting for all genes
                    gene_label = _get_display_gene_name(gene, spe, spe_gene_dict, plotly_opt=False)

                    # Add gene name text
                    plt.annotate(
                        gene_label, (x, y), textcoords="offset points", xytext=(10,10),
                        ha='center', fontsize=12, color='darkblue',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=0.5, alpha=0.7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='darkblue')
                        )
                    
                except ValueError:
                    print(f"Warning: Gene '{gene}' not found in {spe} gene_list_before_preprocessing.")

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA(pointcloud in gene expression space) of {spe}')
            plt.legend()
            plt.grid(True)
            plt.show()



    def plot_pca_highlighting_genes_retained_after_preprocessing_plotly(self, comparison, labeled_genes, spe_gene_dict):
        """plot_pca_highlighting_retained_after_preprocessing_plotly

        Plot expression of all genes, with particular emphasis on genes that remained after preprocessing

        Args:
            - comparison (str): Compared to when all genes were targeted ("all_gene"), or compared to when only transcription factor genes were targeted ("transcription_factors")
            - labeled_genes (dict): A list of genes to be labeled on the scatter plot, grouped by species
            - spe_gene_dict (dict): Gene name notation for each species. ex. spe_gene_dict["mouse"] = "capitalized_italic", spe_gene_dict["human"] = "all_capitalized_italic"

        Outputs:
            - figure

        Examples:
            >>> spe_ot.plot_pca_highlighting_genes_retained_after_preprocessing()
        """
        # Perform PCA
        all_principal_components, explained_variance_ratio_dict = self._perform_pca_to_check_the_consequence_of_preprocessing(comparison)

        # Determine a common axis range from all principal component data
        lower_limit_of_x_axis, upper_limit_of_x_axis, lower_limit_of_y_axis, upper_limit_of_y_axis = (
            self._determine_a_common_axis_range_from_all_principal_component_data(all_principal_components)
        )

        # Get a list of genes before and after preprocessing
        gene_list_before_preprocessing, gene_list_after_preprocessing = (
            self._get_a_list_of_genes_before_and_after_preprocessing(comparison)
        )

        # Create a boolean mask for genes that are in both lists
        is_gene_retained = self._create_a_boolean_mask_for_genes_that_are_in_both_lists(
            gene_list_before_preprocessing, gene_list_after_preprocessing
        )

        for i, spe in enumerate(self.species):
            principal_components = all_principal_components[i]
            print(f"Explain Variance Ratio : {explained_variance_ratio_dict[spe]}")

            # Create DataFrame for easier Plotly handling
            df_pca = pd.DataFrame({
                'PC1': principal_components[:, 0],
                'PC2': principal_components[:, 1],
                'gene': gene_list_before_preprocessing[spe],
                'is_retained': is_gene_retained[spe]
            })

            # Apply common name conversion and formatting for all genes
            df_pca['display_gene_name'] = df_pca['gene'].apply(
                lambda gene: _get_display_gene_name(gene, spe, spe_gene_dict, plotly_opt=True)
            )

            fig = go.Figure()

            # Plot all genes (gray)
            fig.add_trace(go.Scatter(
                x=df_pca['PC1'],
                y=df_pca['PC2'],
                mode='markers',
                marker=dict(color='gray', size=5, opacity=0.7),
                name='All genes',
                text=df_pca['display_gene_name'],  # Use formatted display name
                hoverinfo='text',
                showlegend=True
            ))

            # Plot retained genes (red)
            fig.add_trace(go.Scatter(
                x=df_pca[df_pca['is_retained']]['PC1'],
                y=df_pca[df_pca['is_retained']]['PC2'],
                mode='markers',
                marker=dict(color='red', size=7, opacity=0.8),
                name='Retained genes',
                text=df_pca[df_pca['is_retained']]['display_gene_name'],  # Use formatted display name
                hoverinfo='text',
                showlegend=True
            ))

            # Add labeled genes (blue stars)
            for gene in labeled_genes[spe]:
                try:
                    gene_data = df_pca[df_pca['gene'] == gene]
                    if not gene_data.empty:
                        x = gene_data['PC1'].iloc[0]
                        y = gene_data['PC2'].iloc[0]

                        formatted_gene_label = gene_data['display_gene_name'].iloc[0]

                        fig.add_trace(go.Scatter(
                            x=[x],
                            y=[y],
                            mode='markers+text',
                            marker=dict(symbol='star', size=15, color='blue', line=dict(width=1, color='black')),
                            name=f'Labeled: {formatted_gene_label}',
                            text=[formatted_gene_label],
                            textposition="top center",
                            textfont=dict(size=12, color='darkblue'),
                            hoverinfo='text',
                            hovertext=f"Gene: {formatted_gene_label}<br>PC1: {x:.2f}<br>PC2: {y:.2f}",
                            showlegend=False
                        ))
                except Exception as e:
                    print(f"Warning: Could not label gene '{gene}' in {spe}. Error: {e}")


            fig.update_layout(
                title=f'PCA (pointcloud in gene expression space) of {spe}',
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                hovermode='closest',
                showlegend=True,
                height=700,
                width=1000,
                margin=dict(l=40, r=40, b=40, t=80),
                xaxis=dict(range=[lower_limit_of_x_axis, upper_limit_of_x_axis]),
                yaxis=dict(range=[lower_limit_of_y_axis, upper_limit_of_y_axis])
            )

            fig.show()
    


    def check_read(self, gene):
        """check_read
        
        Function to check data

        Output:
            - figures to check data

        Examples:
            >>> spe_ot.checkread("GATA3")
        """
        gene_figures = []

        for spe in self.species:
            try:
                fig_gene, ax_gene = plt.subplots()
                self.plot_normalized_log_select_preprocessed_masked[spe][gene].plot(ax=ax_gene, marker="o")
                ax_gene.set_title(f"{spe} {gene}")
                ax_gene.set_xticklabels(ax_gene.get_xticklabels(), rotation=90)
                gene_figures.append(fig_gene)
            except KeyError:
                print(f"Warning: Column '{gene}' not found in the Dataframe for species `{spe}`")
            except IndexError:
                print(f"{spe} IndexError")


        for fig in gene_figures:
            plt.show()



    def save_normalized_reads(self):
        """save_normalized_reads

        Function to check gene expression data after preprocessing

        Output:
            - Shape of gene expression data (Polars) after preprocessing

        Examples:
            >>> spe_ot.save_normalized_reads()
        """
        arr = {}
        dfp = {}
        for spe in self.species:
            arr[spe] = np.array(self.adata[spe].obs.index.tolist())
            dfp[spe] = pl.from_numpy(arr[spe], schema=["cells"])

        df0_normalized_log_select_preprocessed_masked = {}
        for spe in self.species:
            df0_normalized_log_select_preprocessed_masked[spe] = pl.from_numpy(
                self.adata[spe].obsm["normalized_log_select_preprocessed_masked"],
                schema=self.hvlabel[spe].tolist(),
            ).with_columns(dfp[spe])


        dfp_normalized_log_select_preprocessed_masked = {}
        for spe in self.species:
            dfp_normalized_log_select_preprocessed_masked[spe] = (
                df0_normalized_log_select_preprocessed_masked[spe].select(
                    pl.col("cells"), pl.exclude("cells")
                )
            )  
            print(spe, df0_normalized_log_select_preprocessed_masked[spe].shape)



    def _generate_maximum_gene_expression_level_histogram_after_cutting_off_poorly_expressed_genes(self):
        """_generate_maximum_gene_expression_level_histogram_after_cutting_off_poorly_expressed_genes
        
        Function to generate a histogram of the maximum expression level of each gene after cutting off poorly expressed genes

        This is used in SpeciesOT class function "data_size_after_preprocessing"
        """
        maxvalue_preprocessed = {}

        fig, ax = plt.subplots()

        for spe in self.species:
            maxvalue_preprocessed[spe] = np.amax(
                self.adata[spe].obsm["normalized_log_select_preprocessed_masked"], axis=0
            )
            
            sns.histplot(maxvalue_preprocessed[spe], ax=ax, bins=100, kde=True)
            ax.set_title(
                "Histogram of the maximum expression level of each gene\n after cutting off poorly expressed genes"
            )
            ax.set_xlabel("Maximum expression level")
            ax.axvline(x=self.threshold, color="red")
            if (
                self.data_option == "dataset2"
            ):
                ax.text(self.threshold * 1.05, 100, f"log2(RP100K+1)={self.threshold}")
                ax.set_ylim(0, 120)
                ax.set_xlim(0, 8)
            elif (
                self.data_option == "dataset1"
            ):
                ax.text(self.threshold * 1.05, 100, f"log2(RPM+1)={self.threshold}")
                ax.set_ylim(0, 120)
                ax.set_xlim(0, 12)
            else:
                print(self.data_option)
            
        ax.set_ylabel("Frequency")
        ax.legend(self.species_labels)

        plt.show()



    def data_size_after_preprocessing(self):
        """data_size_after_preprocessing
        
        Function to check data after normalization and preprocessing

        Outputs:
            - Shape of adata after normalization and preprocessing
            - Number of genes with all zero expression
            - Histogram of the maximum expression level of each gene after cutting off poorly expressed genes

        Examples:
            >>> spe_ot.data_size_after_preprocessing()
        """
        print("Shape of adata after normalization and preprocessing")
        for spe in self.species:
            print(spe, self.adata[spe].obsm["normalized_log_select_preprocessed_masked"].shape)

        print("# genes with all zero expression")
        for spe in self.species:
            print(
                spe,
                np.sum(
                    np.all(
                        self.adata[spe].obsm["normalized_log_select_preprocessed_masked"] == 0,
                        axis=0,
                    )
                ),
            )

        # Histogram of the maximum expression level of each gene after cutting off poorly expressed genes
        self._generate_maximum_gene_expression_level_histogram_after_cutting_off_poorly_expressed_genes()



    def pca(self):
        """pca
        
        Function to perform principal component analysis
        
        Returns:
            - self

        Examples:
            >>> spe_ot = spe_ot.pca()
        """
        if (
            self.data_option == "dataset1"
        ):
            self.pca_dim = 2
        elif (
            self.data_option == "dataset2"
        ):
            self.pca_dim = 2
        elif self.data_option == "custom":
            self.pca_dim = 2
        else:
            self.pca_dim = 10

        self.pca_embedding = {}
        self.explained_variance_ratio = {}

        pca = sklearn.decomposition.PCA(n_components=self.pca_dim)

        for spe in self.species:
            # Fit PCA on the dataset
            self.pca_embedding[spe] = pca.fit_transform(
                self.adata[spe]
                .obsm["normalized_log_select_preprocessed_masked"]
                .T
            )
            # Retrieve the explained variance ratio for the first nine components
            self.explained_variance_ratio[spe] = pca.explained_variance_ratio_

        return self            
    


    def _plot_explained_variance_ratio_by_pca_components(self, label_fontsize=12, title_fontsize=14, text_fontsize=10):
        """_plot_explained_variance_ratio_by_pca_components

        Function to plot explained variance ratio by pca components
        
        This is used in SpeciesoT class function "visualize_pca"
        """
        for spe in self.species:
            fig, ax = plt.subplots(figsize=(10,6))

            bars = plt.bar(range(1, self.pca_dim + 1), self.explained_variance_ratio[spe])
            ax.set_xticks(range(1, self.pca_dim + 1), [f"PC{i}" for i in range(1, self.pca_dim + 1)])
            ax.set_xlabel("Principal Components", fontsize=label_fontsize)
            ax.set_ylabel("Explained Variance Ratio", fontsize=label_fontsize)
            ax.set_title(f"Explained Variance by PCA Components ( {spe} )", fontsize = title_fontsize)

            for bar in bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval,
                    f"{yval*100:.2f}%",
                    va="bottom",
                    ha="center",
                    fontsize=text_fontsize,
                )

            plt.show()



    def _plot_point_cloud_in_gene_expression_space(self):
        """_plot_point_cloud_in_gene_expression_space

        Function to plot point cloud in gene expression space
        
        This is used in SpeciesoT class function "visualize_pca"
        """
        for spe in self.species:
            fig, ax = plt.subplots()
            ax.scatter(
                self.pca_embedding[spe][:, 0],
                self.pca_embedding[spe][:, 1],
                label=spe,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA(pointcloud in gene expression space) of {spe}")
            ax.legend()
            plt.show()



    def _plot_merged_point_cloud_in_gene_expression_space(self, spe1, spe2):
        """_plot_merged_point_cloud_in_gene_expression_space

        Function to plot merged point cloud of two species in gene expression space
        
        This is used in SpeciesoT class function "visualize_pca"
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.scatter(
            self.pca_embedding[spe1][:, 0],
            self.pca_embedding[spe1][:, 1],
            label=spe1,
        )
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_title(f"PCA(pointcloud in gene expression space) of {spe1}")
        ax1.legend()

        ax2.scatter(
            self.pca_embedding[spe2][:, 0],
            self.pca_embedding[spe2][:, 1],
            label=spe2,
        )
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title(f"PCA(pointcloud in gene expression space) of {spe2}")
        ax2.legend()

        plt.show()



    def _plot_merged_point_cloud_in_gene_expression_space2(self, spe1, spe2):
        """_plot_merged_point_cloud_in_gene_expression_space2

        Function to plot merged point cloud of two species in gene expression space
        
        This is used in SpeciesoT class function "visualize_pca"
        """
        # Create subplots: 1 row, 2 columns
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"{spe1}",
                f"{spe2}",
            ),
        )

        # Adding scatter plot for the first species in the first subplot with labels
        fig.add_trace(
            go.Scatter(
                x=self.pca_embedding[spe1][:, 0],
                y=self.pca_embedding[spe1][:, 1],
                mode="markers+text",
                name=spe1,
                marker=dict(color="red"),
                text=self.hvlabel[spe1],  
                textposition="top center",
            ),
            row=1,
            col=1,
        )

        # Adding scatter plot for the second species in the second subplot with labels
        fig.add_trace(
            go.Scatter(
                x=self.pca_embedding[spe2][:, 0],
                y=self.pca_embedding[spe2][:, 1],
                mode="markers+text",
                name=spe2,
                marker=dict(color="blue"),
                text=self.hvlabel[spe2],  # Add labels for each point
                textposition="top center",
            ),
            row=1,
            col=2,
        )

        # Update xaxis and yaxis properties for both subplots
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="PC1", row=1, col=2)
        fig.update_yaxes(title_text="PC2", row=1, col=2)

        # Update the layout and display the figure
        fig.update_layout(
            height=600,
            width=1200,
            title_text="PCA(pointcloud in gene expression space)",
        )
        
        plt.show()



    def _set_target_labels(self, opt):
        """_set_target_labels
        
        Function to set target_labels

        This is used in SpeciesoT class function "_plot_merged_point_cloud_in_masked_gene_expression_space", "_plot_merged_point_cloud_in_3d_gene_expression_space"
        """
        target_labels = {}

        if opt == "3_genes":
            for spe in self.species:
                if spe == self.species[0]:
                    target_labels[spe] = [
                        "EOMES",
                        "PRDM1",
                        "SOX17",
                    ]
                elif spe == self.species[-1]:
                    target_labels[spe] = [
                        "TBXT",
                        "PRDM1",
                        "SALL4",
                    ]
                else:
                    target_labels[spe] = []

        else:
            for spe in self.species:
                if spe == self.species[0]:
                    target_labels[spe] = [
                        "GATA3",
                        "GATA2",
                        "EOMES",
                        "SOX17",
                        "PRDM1",
                        "TFAP2C",
                    ]
                elif spe == self.species[1]:
                    target_labels[spe] = [
                        "GATA3",
                        "GATA2",
                        "EOMES",
                        "SOX17",
                        "PRDM1",
                        "TFAP2C",
                    ]
                else:
                    target_labels[spe] = []

        return target_labels



    def _plot_merged_point_cloud_in_masked_gene_expression_space(self, spe1, spe2, target_labels_opt="5_genes"):
        """_plot_merged_point_cloud_in_masked_gene_expression_space

        Function to plot merged point cloud of two species in gene expression space
        
        This is used in SpeciesoT class function "visualize_pca"
        """
        target_labels = self._set_target_labels(target_labels_opt)

        hvlabel_masked = {}
        for spe in self.species:
            hvlabel_masked[spe] = [
                label if label in target_labels[spe] else None for label in self.hvlabel[spe]
            ]

        # Create subplots: 1 row, 2 columns
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"{spe1}",
                f"{spe2}",
            ),
        )

        # Adding scatter plot for the first species in the first subplot with labels
        fig.add_trace(
            go.Scatter(
                x=self.pca_embedding[spe1][:, 0],
                y=self.pca_embedding[spe1][:, 1],
                mode="markers+text",
                name=spe1,
                marker=dict(color="red"),
                text=hvlabel_masked[spe1],  # Add labels for each point
                textposition="top center",
            ),
            row=1,
            col=1,
        )

        # Adding scatter plot for the second species in the second subplot with labels
        fig.add_trace(
            go.Scatter(
                x=self.pca_embedding[spe2][:, 0],
                y=self.pca_embedding[spe2][:, 1],
                mode="markers+text",
                name=spe2,
                marker=dict(color="blue"),
                text=hvlabel_masked[spe2],  # Add labels for each point
                textposition="top center",
            ),
            row=1,
            col=2,
        )

        # Update xaxis and yaxis properties for both subplots
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="PC1", row=1, col=2)
        fig.update_yaxes(title_text="PC2", row=1, col=2)

        # Update the layout and display the figure
        fig.update_layout(
            height=600,
            width=1200,
            title_text="PCA(pointcloud in gene expression space)",
        )

        plt.show()    



    def _plot_merged_point_cloud_in_3d_gene_expression_space(self, spe1, spe2, color_opt):
        """_plot_merged_point_cloud_in_3d_gene_expression_space

        Function to plot merged point cloud of two species in 3-dim gene expression space
        
        This is used in SpeciesoT class function "visualize_pca"
        """
        target_labels = self._set_target_labels("5_genes")

        # Update the layout of the plot with proper axis labels
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=(f"{spe1}", f"{spe2}"),
        )

        # Update the layout of the plot with proper axis labels
        if color_opt == "assign":
            colors_spe1 = assign_colors(self.hvlabel[spe1], target_labels[spe1], "grey")
        else:
            colors_spe1 = "red"

        fig.add_trace(
            go.Scatter3d(
                x=self.pca_embedding[spe1][:, 0],
                y=self.pca_embedding[spe1][:, 1],
                z=self.pca_embedding[spe1][:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors_spe1,
                ),
                name=spe1,
                text=self.hvlabel[spe1],  # Add hover labels for each point
                hoverinfo="text",  # Customize hover to show only the text
            ),
            row=1,
            col=1,
        )

        # Update the layout of the plot with proper axis labels
        if color_opt == "assign":
            colors_spe2 = assign_colors(self.hvlabel[spe2], target_labels[spe2], "grey")
        else:
            colors_spe2 = "blue"

        fig.add_trace(
            go.Scatter3d(
                x=self.pca_embedding[spe2][:, 0],
                y=self.pca_embedding[spe2][:, 1],
                z=self.pca_embedding[spe2][:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors_spe2,
                ),
                name=spe2,
                text=self.hvlabel[spe2],  # Add hover labels for each point
                hoverinfo="text",  # Customize hover to show only the text
            ),
            row=1,
            col=2,
        )
    
        # Update the layout of the plot with proper axis labels
        fig.update_layout(height=600, width=1200, title_text="PCA(pointclouds)")

        fig.update_scenes(
            xaxis_title_text="PC1", yaxis_title_text="PC2", zaxis_title_text="PC3"
        )

        plt.show()       



    def visualize_pca(self, spe1="", spe2=""):
        """visualize_pca
        
        Function to visualize data after principal component analysis

        Outputs:
            - Explained Variance by PCA Components
            - Pointcloud in gene expression space

        Examples:
            >>> spe_ot.visualize_pca()
        """
        # Explained Variance by PCA Components
        self._plot_explained_variance_ratio_by_pca_components()

        # Pointcloud in gene expression space
        self._plot_point_cloud_in_gene_expression_space()

        # Choose two species
        if spe1 == "":
            spe1 = self.species[0]
        if spe2 == "":
            spe2 = self.species[1]

        # Merged point cloud in gene expression space
        self._plot_merged_point_cloud_in_gene_expression_space(spe1, spe2)

        # Another version of merged point cloud in gene expression space
        self._plot_merged_point_cloud_in_gene_expression_space2(spe1, spe2)

        # Merged point cloud in masked gene expression space
        self._plot_merged_point_cloud_in_masked_gene_expression_space(spe1, spe2, target_labels_opt="3_genes")

        # Another version of merged point cloud in masked gene expression space
        self._plot_merged_point_cloud_in_masked_gene_expression_space(spe1, spe2, target_labels_opt="5_genes")

        if self.pca_dim > 2:
            # Merged point cloud in 3-dim gene expression space
            self._plot_merged_point_cloud_in_3d_gene_expression_space(spe1, spe2, color_opt="red_blue")

            # Another version of merged point cloud in 3-dim gene expression space
            self._plot_merged_point_cloud_in_3d_gene_expression_space(spe1, spe2, color_opt="assign")



    def calculate_gene_distance_matrix(self):
        """calculate_gene_distance_matrix
        
        Function to calculate the distance between genes for each species and put them into a matrix

        Returns:
            - self

        Examples:
            >>> spe_ot = spe_ot.calculate_dismat()
        """
        if self.metric_option == "euclidean":
            dis_mat_ori = {}
            self.dis_mat = {}
            for spe in self.species:
                if self.dismat_option == "original":
                    mat = (
                        self.adata[spe].obsm["normalized_log_select_preprocessed_masked"].T
                    )  
                elif self.dismat_option == "pca":
                    mat = self.pca_embedding[spe]  

                dis_mat_ori[spe] = _weighted_cdist(mat)
                self.dis_mat[spe] = dis_mat_ori[spe] / np.mean(
                    dis_mat_ori[spe]
                )  

        return self
    


    def check_gene_distance_matrix(self, raw_return_opt=False):
        """check_gene_dismat
        
        Function to check the calculated distance matrix

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Output:
            - sns.heatmap(df_dis_mat[spe], ax=ax_dismat)
            - spe, np.mean(df_dis_mat[spe])
            - spe, np.var(df_dis_mat[spe].to_numpy().flatten())

        Examples:
            >>> spe_ot.check_dismat()
        """
        df_dis_mat = {}

        for spe in self.species:
            df_dis_mat[spe] = pd.DataFrame(
                self.dis_mat[spe], index=self.hvlabel[spe], columns=self.hvlabel[spe]
            )
            print(spe, df_dis_mat[spe].shape)

            fig, ax = plt.subplots()
            sns.heatmap(df_dis_mat[spe], ax=ax)
            ax.set_title(f"Transriptome distribution of {spe}")
            plt.show()

        print("Check mean of gene distance matrices")
        for spe in self.species:
            print(spe, np.mean(df_dis_mat[spe]))

        print("Check variance of gene distance matrices")
        for spe in self.species:
           print(spe, np.var(df_dis_mat[spe].to_numpy().flatten()))

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print("The number of object that can be returned is 1")
            print("The returned object (`df_dis_mat`) is a dict")
            print("The key of `df_dis_mat` is `self.species`")
            print("The value of `df_dismat` is pd.DataFrame")



    def gromov_wasserstein_ot(self):
        """gromov_wasserstein_ot
        
        Function for Gromov-Wasserstein Optimal Transport calculations

        Returns:
            - self

        Output:
            - Whether the results converged for a given epsilon.
            - Best epsilon for convergence

        Examples:
            >>> spe_ot = speciesot.gromov_wasserstein_ot()
        """
        if self.gwot_option == "min":
            # Initialize geometries for each species (fix iteration order)
            geometries = {
                spe: _compute_geometries([jnp.array(self.dis_mat[spe])])[0] for spe in self.species
            }
            initial_high_epsilon = self.high_epsilon

            # Initial check for convergence with high_epsilon
            converged, coupling_matrices, gw_dis_mat = _check_convergence(
                initial_high_epsilon, geometries, self.iterations
            )
            _print_convergence_results(
                initial_high_epsilon, converged
            )  # Print initial results

            if converged:
                # Binary search to find the smallest epsilon that still converges
                lo_eps, hi_eps = self.low_epsilon, initial_high_epsilon

                while hi_eps - lo_eps > self.threshold_eps:
                    mid_epsilon = (hi_eps + lo_eps) / 2.0
                    temp_converged, temp_coupling_matrices, temp_gw_dis_mat = _check_convergence(
                        mid_epsilon, geometries, self.iterations
                    )
                    _print_convergence_results(
                        mid_epsilon, temp_converged
                    )  # Print results for each mid_epsilon

                    if temp_converged:
                        hi_eps = mid_epsilon
                    else:
                        lo_eps = mid_epsilon

                # Reflect narrowed bounds
                self.low_epsilon, self.high_epsilon = lo_eps, hi_eps

                # >>> Recompute at best_epsilon and SAVE the artifacts <<<
                self.best_epsilon = self.high_epsilon
                final_converged, self.coupling_matrices, self.gw_dis_mat = _check_convergence(
                    self.best_epsilon, geometries, self.iterations
                )
                # final_converged should be True by construction

            else:
                print("Adjust your epsilon range.")
                self.best_epsilon = None

            if self.best_epsilon is not None:
                print(
                    f"Best epsilon for convergence: {self.best_epsilon} (threshold = {self.threshold_eps})."
                )
            else:
                print("Convergence not achieved within the given epsilon range.")

        
        elif self.gwot_option == "fixed":
            # Initialize geometries for each species (fix iteration order)
            geometries = {
                spe: _compute_geometries([jnp.array(self.dis_mat[spe])])[0] for spe in self.species
            }
            initial_high_epsilon = self.high_epsilon

            # Check for convergence with high_epsilon
            converged, self.coupling_matrices, self.gw_dis_mat = _check_convergence(
                initial_high_epsilon, geometries, self.iterations
            )

            if converged:
                self.best_epsilon = self.high_epsilon
                _print_convergence_results(
                    self.best_epsilon, converged
                )
            else:
                print("GWOT solver did not converge!")


        # Convert to pandas DataFrame
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



    def _visualize_gw_distance_matrix(self):
        """_visualize_gw_distance_matrix
        
        Function to visualize GW distance matrix

        This is used in SpeciesOT class function "check_results"
        """
        fig, ax = plt.subplots()
        sns.heatmap(
            self.gw_dis_mat, ax=ax, cmap="viridis_r", annot=True,
            xticklabels=self.species, yticklabels=self.species
        )
        ax.set_title(f"GW distance (threshold = {self.best_epsilon})")
        plt.show()



    def _visualize_optimal_transport_plan(self, species_pair):
        """_visualize_gw_distance_matrix
        
        Function to visualize OTP of self.species_pairs[1]

        This is used in SpeciesOT class function "check_results"
        """
        fig, ax = plt.subplots()
        sns.heatmap(self.df_coupling_matrices[species_pair], vmax=3.5e-6, cmap="viridis_r")
        ax.set_title(f"Optimal transport plan {species_pair}")
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.show()


    
    def check_optimal_transport_results(self, species_pair="", raw_return_opt=False):
        """check_optimal_transport_results

        Function to check calculated GW distance and OTP

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Output:
            - GW distance matrix
            - Optimal transport plan (self.species_pairs[1])

        Examples:
            >>> spe-ot.check_results()
        """
        # GW distance matrix
        self._visualize_gw_distance_matrix()

        if raw_return_opt:
            print("The returned object (`self.gw_dis_mat`) is jnp.ndarray")


        # Optimal transport plan (self.species_pairs[1])
        if species_pair == "":
            species_pair = self.species_pairs[1]
        self._visualize_optimal_transport_plan(species_pair)

        if raw_return_opt:
            print("The returned object (`self.df_coupling_matrices[species_pair]`) is pd.DataFrame")


        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print("The number of objects that can be returned is 2")
            return self.gw_dis_mat, self.df_coupling_matrices[species_pair]

    

    def filter_otp(self, raw_return_opt=False):
        """filter_otp
        
        Function to plot OTP after thresholding the top 5%

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Outputs:
            - Heatmap of OTP after thresholding the top 5%

        Examples:
            >>> spe_ot.filter_otp()
        """
        filtering_value = 99

        filtered_otp = {}
        for key in self.species_pairs:
            # Calculate the 5th percentile of all values in the DataFrame using numpy
            percentile_value = np.percentile(
                self.df_coupling_matrices[key].values.flatten(), filtering_value
            )

            # Filter the DataFrame: Keep values greater than or equal to the 5th percentile, and set others to NaN
            filtered_otp[key] = self.df_coupling_matrices[key].where(
                self.df_coupling_matrices[key] >= percentile_value
            )

            # Plotting the heatmap
            fig, ax = plt.subplots()
            sns.heatmap(filtered_otp[key], ax=ax, vmax=3.5e-6, cmap="viridis_r")
            ax.set_title(
                "Filtered optimal transport plan "
                + key
                + "\n(percentile_value = "
                + str(percentile_value)
                + ")"
            )
            ax.set_ylabel("")
            ax.set_xlabel("")

        plt.show()

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print(f"The number of object that can be returned is 1")
            print("The returned object (`filtered_otp`) is a dict")
            print("The key of `filterd_otp` is `self.species_pairs`")
            print("The value of `filtered_otp` is pd.DataFrame")



    def _generate_heatmap_of_gw_distance_matrix(self):
        """_generate_heatmap_of_gw_distance_matrix
        
        Function to generate a heatmap of GW distance matrix

        This is used in SpeciesOT class function "dendrogram"
        """
        fig, ax = plt.subplots()
        sns.heatmap(self.gw_dis_mat, ax=ax, annot=True, fmt=".3f")
        ax.set_xticklabels(self.species, rotation=0)
        ax.set_yticklabels(self.species, rotation=0)
        plt.title("GW distance matrix")    
        plt.show()



    def _set_linkage_mat_and_labels(self, df, reverse_opt):
        """_set_gw_dismat_and_labels_for_dendrogram
        
        Function to perform hierarchical clustering for dendrogram

        This is used in SpeciesOT class function "dendrogram"
        """
        if reverse_opt:
            # Reverse the input matrix both row-wise and column-wise
            reversed_mat = np.flipud(np.fliplr(df))
            # Perform hierarchical clustering on the reversed matrix
            linked = linkage(pdist(reversed_mat), "ward")
            # Define your labels
            labels = self.species_labels[::-1]  # Reversed labels to match the reversed matrix
        else:
            # Perform hierarchical clustering
            linked = linkage(pdist(df), "ward")
            # Define your labels
            labels = self.species_labels

        return linked, labels



    def _plot_dendrogram_from_linkage_matrix_and_labels(
            self, linked, labels, xlabel="Regularized GW cost", reverse_opt=False
        ):
        """_plot_dendrogram_from_linkage_matrix_and_labels
        
        Function to plot dendrogram based on a linkage matrix and labels

        This is used in SpeciesOT class function "dendrogram"
        """        
        # Define the custom color for the dendrogram
        custom_color = "#ff0000"

        fig, ax = plt.subplots(figsize=(6, 3))

        dendrogram(
            linked,
            orientation="left",
            ax=ax,
            labels=labels,
            distance_sort="descending",
            show_leaf_counts=True,
            color_threshold=np.inf,  # Set to infinity to apply the color to all links
            link_color_func=lambda k: custom_color,  # Apply custom color
        )

        if reverse_opt:
            ax.invert_yaxis()  # Invert the y-axis to reverse the dendrogram vertically

        ax.set_title(
            "{}_{}_{}_{}_{}".format(
                self.data_option, self.gene_option, self.mask_option, self.best_epsilon, self.dismat_option
            )
        )

        xlabel = xlabel
        ax.set_xlabel(xlabel)

        plt.show()



    def _calculate_epsilon_entropy(self):
        """_calculate_epsilon_entropy
        
        Function to calculate epsilon entropy

        This is used in SpeciesOT class function "dendrogram", "_calculate_entropy_gw_distance"        
        """
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
    


    def _visualize_epsilon_entropy(self, hyp_entropy_mat):
        """_visualize_epsilon_entropy
        
        Function to visualize epsilon entropy

        This is used in SpeciesOT class function "dendrogram"        
        """
        fig, ax = plt.subplots()
        ax = sns.heatmap(hyp_entropy_mat, ax=ax, annot=True, fmt=".3f")
        ax.set_xticklabels(self.species, rotation=0)
        ax.set_yticklabels(self.species, rotation=0)
        ax.set_title("epsilon * entropy")
        plt.show()



    def _calculate_gw_cost(self, hyp_entropy_mat):
        """_calculate_gw_cost
        
        Function to calculate GW cost

        This is used in SpeciesOT class function "dendrogram", "_calculate_entropy_gw_distance"       
        """
        entropy_jax_array = jnp.array(hyp_entropy_mat.values)
        gw_cost_mat = self.gw_dis_mat + entropy_jax_array

        # Convert the resulting JAX array back to a pandas DataFrame
        gw_cost_df = pd.DataFrame(gw_cost_mat, index=self.species, columns=self.species)

        return gw_cost_df
    


    def _visualize_gw_cost(self, gw_cost_df):
        """__visualize_gw_cost
        
        Function to visualize GW cost

        This is used in SpeciesOT class function "dendrogram"        
        """
        fig, ax = plt.subplots()
        sns.heatmap(gw_cost_df, ax=ax, annot=True, fmt=".3f")
        ax.set_xticklabels(self.species, rotation=0)
        ax.set_yticklabels(self.species, rotation=0)
        ax.set_title("GW cost")
        plt.show()



    def _calculate_entropy_gw_distance(self):
        """_calculate_entropy_gw_distance
        
        Function to calculate Entropy GW distance

        This is used in SpeciesOT class function "dendrogram", "plot_transcriptomic_discrepancy", "outputs_for_paper"
        """
        hyp_entropy_mat = self._calculate_epsilon_entropy()

        gw_cost_df = self._calculate_gw_cost(hyp_entropy_mat)

        entropy_gw_distance = np.sqrt(gw_cost_df)

        return entropy_gw_distance
    


    def _visualize_entropy_gw_distance(self, entropy_gw_distance):
        """_visualize_entropy_gw_distance(self, entropy_gw_distance)
        
        Function to visualize Entropy GW distance

        This is used in SpeciesOT class function "dendrogram", "plot_transcriptomic_discrepancy", "outputs_for_paper"        
        """
        ax = sns.heatmap(entropy_gw_distance, annot=True, fmt=".3f")
        if self.data_option == "dataset2":
            ax.set_xticklabels(self.species_labels, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(self.species_labels, rotation=0)
        ax.set_yticklabels(self.species_labels, rotation=0)
        plt.title("Entropy GW distance")
        plt.show()



    def dendrogram(self , opt="all", raw_return_opt=False):
        """dendrogram
        
        Function to calculate dendrogram

        Args:
            opt (string): "one" or "all"
            Parameters to determine the number of figs to output.
            Default to "all".
            raw_return_opt (bool) : Whether to return raw data

        Outputs:
            - Heatmap of GW distance matrix
            - Dendrogram derived from GW distance matrix 
            - Reversed dendrogram derived from GW distance matrix
            - Heatmap of epsilon*entropy
            - Heatmap of GW cost
            - Dendrogram derived from GW cost
            - Reversed dendrogram derived from GW cost
            - Heatmap of entropy GW distance
            - Dendrogram derived from entropy GW distance
            - Reversed dendrogram derived from entropy GW distance

        Examples:
            >>> spe_ot.dendrogram()
        """    
        returnable_object_type_list = []

        if opt == "all":
            # Heatmap of GW distance matrix
            self._generate_heatmap_of_gw_distance_matrix()

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "jnp_heatmap", "self.gw_dis_mat"
            )


            # Dendrogram derived from GW distance matrix 
            gw_dismat_linked, labels = \
                self._set_linkage_mat_and_labels(self.gw_dis_mat, reverse_opt=False)
            self._plot_dendrogram_from_linkage_matrix_and_labels(gw_dismat_linked, labels, reverse_opt=False)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "gw_dismat_linked"
            )


            # Reversed dendrogram derived from GW distance matrix
            gw_dismat_reversed_linked, reversed_labels = \
                self._set_linkage_mat_and_labels(self.gw_dis_mat, reverse_opt=True)
            self._plot_dendrogram_from_linkage_matrix_and_labels(gw_dismat_reversed_linked, reversed_labels, reverse_opt=True)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "gw_dismat_reversed_linked"
            )


            # epsilon*entropy
            hyp_entropy_mat = self._calculate_epsilon_entropy()
            self._visualize_epsilon_entropy(hyp_entropy_mat)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "heatmap", "hyp_entropy_mat"
            )


            # Heatmap of GW cost
            gw_cost_df = self._calculate_gw_cost(hyp_entropy_mat)
            self._visualize_gw_cost(gw_cost_df)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "heatmap", "gw_cost_df"
            )


            # Dendrogram derived from GW cost
            gw_cost_linked, labels = \
                self._set_linkage_mat_and_labels(gw_cost_df, reverse_opt=False)
            self._plot_dendrogram_from_linkage_matrix_and_labels(gw_cost_linked, labels, reverse_opt=False)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "gw_cost_linked"
            )


            # Reversed dendrogram derived from regularized GW cost
            gw_cost_reversed_linked, reversed_labels = \
                self._set_linkage_mat_and_labels(gw_cost_df, reverse_opt=True)
            self._plot_dendrogram_from_linkage_matrix_and_labels(gw_cost_reversed_linked, reversed_labels, reverse_opt=True)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "gw_cost_reversed_linked"
            )


            # entropy GW distance
            entropy_gw_distance = np.sqrt(gw_cost_df)
            self._visualize_entropy_gw_distance(entropy_gw_distance)

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "heatmap", "entropy_gw_distance"
            )


            # Dendrogram derived from entropy GW distance
            entropy_gw_dismat_linked, labels = \
                self._set_linkage_mat_and_labels(entropy_gw_distance, reverse_opt=False)
            
            self._plot_dendrogram_from_linkage_matrix_and_labels(
                entropy_gw_dismat_linked, labels, xlabel="Entropy GW distance", reverse_opt=False
            )

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "entropy_gw_dismat_linked"
            )


            # Reversed dendrogram derived from entropy GW distance
            entropy_gw_dismat_reversed_linked, labels = \
                self._set_linkage_mat_and_labels(entropy_gw_distance, reverse_opt=True)
            
            self._plot_dendrogram_from_linkage_matrix_and_labels(
                entropy_gw_dismat_reversed_linked, labels, xlabel="Entropy GW distance", reverse_opt=True
            )

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "entropy_gw_dismat_reversed_linked"
            )


            # Return raw data
            if raw_return_opt:
                print("# Raw data summary")
                print(f"The number of objects that can be returned is {len(returnable_object_type_list)}")
                return self.gw_dis_mat, gw_dismat_linked, gw_dismat_reversed_linked, \
                    entropy_gw_dismat_reversed_linked, hyp_entropy_mat, gw_cost_df, \
                    gw_cost_linked, gw_cost_reversed_linked, entropy_gw_distance, \
                    entropy_gw_dismat_linked, entropy_gw_dismat_reversed_linked
            

        else:
            # Calculate entropy GW distance
            entropy_gw_distance = self._calculate_entropy_gw_distance()


            # Reversed dendrogram derived vrom entropy GW distance
            entropy_gw_dismat_reversed_linked, labels = \
                self._set_linkage_mat_and_labels(entropy_gw_distance, reverse_opt=True)
            
            self._plot_dendrogram_from_linkage_matrix_and_labels(
                entropy_gw_dismat_reversed_linked, labels, xlabel="Entropy GW distance", reverse_opt=True
            )

            returnable_object_type_list = _append_to_returnable_object_type_list(
                raw_return_opt, returnable_object_type_list, "dendrogram", "entropy_gw_dismat_reversed_linked"
            )

            # Return raw data
            if raw_return_opt:
                print("# Raw data summary")
                print(f"The number of objects that can be returned is {len(returnable_object_type_list)}")
                return entropy_gw_dismat_reversed_linked



    def _calculate_sinkhorn_entropy_gw_distance(self, entropy_gw_distance):
        """_calculate_sinkhorn_entropy_gw_distance
        
        Function to calculate Sinkhorn entropy GW distance (= Transcriptomic discrepancy array)

        This is used in SpeciesOT class function "plot_transcriptomic_discrepancy", "outputs_for_paper"        
        """
        df = entropy_gw_distance

        # Create a matrix of diagonal values repeated
        diag = np.diag(df)

        # Use broadcasting to compute the required transformation
        result = df - 0.5 * (diag[:, np.newaxis] + diag[np.newaxis, :])  # Transcriptomic distance array

        # Symmetrization
        sym_result = (result + result.T) / 2  # Transcriptomic discrepancy array

        # Convert to DataFrame
        sinkhorn_entropy_gw_distance = pd.DataFrame(sym_result, columns=df.columns, index=df.index)

        return sinkhorn_entropy_gw_distance
    
    

    def _visualize_sinkhorn_entropy_gw_distance(self, sinkhorn_entropy_gw_distance):
        """_visualize_entropy_gw_distance
        
        Function to visualize Sinkhorn entropy GW distance

        This is used in SpeciesOT class function "plot_transcriptomic_discrepancy", "outputs_for_paper"        
        """
        ax = sns.heatmap(sinkhorn_entropy_gw_distance, annot=True, fmt=".3f")
        if self.data_option == "dataset2":
            ax.set_xticklabels(self.species_labels, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(self.species_labels, rotation=0)
        ax.set_yticklabels(self.species_labels, rotation=0)
        plt.title("Transcriptomic discrepancy array")
        plt.show()



    def _set_linkage_and_labels_for_transcriptomic_discrepancy_tree(self, sinkhorn_entropy_gw_distance):
        """_set_linkage_and_labels_for_transcriptomic_discrepancy_tree
        
        Function to set linkage and labels for transcriptomic discrepancy tree

        This is used in SpeciesOT class function "plot_transcriptomic_discrepancy", "outputs_for_paper"   
        """
        # Define your labels
        labels = self.species_labels

        if self.data_option == "dataset1":
            # Reverse the input distance matrix both row-wise and column-wise
            reversed_sinkhorn_entropy_gw_distance = np.flipud(
                np.fliplr(sinkhorn_entropy_gw_distance)
            )
            # linked_sinkhorn = linkage(pdist(reversed_sinkhorn_entropy_gw_distance), "ward") # incorect former definition
            condensed = squareform(reversed_sinkhorn_entropy_gw_distance)

            # Reversed labels to match the reversed matrix
            labels = labels[::-1]
        else:
            # linked_sinkhorn = linkage(pdist(sinkhorn_entropy_gw_distance), "ward")
            condensed = squareform(sinkhorn_entropy_gw_distance)

        linked_sinkhorn = linkage(condensed, "single")

        X = np.array(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
            ]
        )
        ordered_linkage = optimal_leaf_ordering(linked_sinkhorn, pdist(X))

        return ordered_linkage, labels



    def _generate_transcriptomic_discrepancy_tree(self, ordered_linkage, labels):
        """_generate_transcriptomic_discrepancy_tree
        
        Function to generate transcriptomic discrepancy tree

        This is used in SpeciesOT class function "entropy_gw_dendrogram", "outputs_for_paper"        
        """
        # Define the custom color for the dendrogram
        # custom_color = "#ff0000"
        custom_color = "#000000"

        plt.figure(figsize=(4.5, 6))
        dendrogram(
            ordered_linkage,
            orientation="left",
            labels=labels,
            # distance_sort="descending",
            show_leaf_counts=True,
            color_threshold=np.inf,
            link_color_func=lambda k: custom_color,
        )

        ax = plt.gca()
        ax.set_xlabel("Transcriptomic discrepancy")
        ax.set_xlabel("")
        plt.title("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        plt.title("Transcriptomic discrepancy tree")
        plt.xlabel("Transcriptomic discrepancy")
        plt.show()       



    def plot_transcriptomic_discrepancy(self, raw_return_opt=False):
        """plot_transcriptomic_discrepancy
        
        Function to calculate entropy GW distance and Sinkhorn entropy GW distance and dendrogram

        Outputs:
            - Entropy GW distance
            - Sinkhorn entropy GW distance
            - Transcriptomic discrepancy tree

        Examples:
            >>> spe_ot.plot_transcriptomic_discrepancy()
        """
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



    def intersection_of_gene_lists(self, raw_return_opt=False):
        """intersection_of_gene_lists
        
        Function to creat a heatmap of common genes

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Outputs:
            - Heatmap of intersection of gene lists

        Examples:
            - spe_ot.intersection_of_gene_lists()
        """
        survived_genes_1 = {}
        survived_genes_2 = {}
        intersection_genes = {}

        for key in self.species_pairs:
            survived_genes_1[key] = set(self.df_coupling_matrices_normalized[key].index)
            survived_genes_2[key] = set(self.df_coupling_matrices_normalized[key].columns)
            intersection_genes[key] = survived_genes_1[key].intersection(survived_genes_2[key])        

        # Initialize an empty dictionary to store the data for the new DataFrame
        data_dict = {}

        # Loop through the Species_pairs
        for key in self.species_pairs:
            index = key.split("_")[0]
            column = key.split("_")[1]
            value = len(intersection_genes[key])

            # If the index (row) doesn't exist yet, create it
            if index not in data_dict:
                data_dict[index] = {}

            # Assign the value for the specific column (Species_pair)
            data_dict[index][column] = value

        # Create the DataFrame from the dictionary
        common_genes_table = pd.DataFrame.from_dict(data_dict, orient="index")
        common_genes_table = common_genes_table.fillna(0)

        sns.heatmap(common_genes_table, annot=True, fmt="g", cmap="viridis", linewidths=0.5)
        plt.title("Intersection of Gene Lists")
        plt.xlabel("")
        plt.ylabel("")
        plt.show()

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print("The number of objects that can be returned is 1")
            print("The returned object (`common_genes_table`) is pd.DataFrame")
            return common_genes_table



    def normalize_otp(self):
        """normalize_otp
        
        Function to normalize optimal transport plans (= gene-to-gene corresponding) and create Dataframes with the most corresponding genes for each gene in order

        Since the sum of all elements is difficult to understand with 1, we multiplied otp by the number of rows x 100.

        cf. other normalization formulas for optimal transport plans, e.g. long transform

        Returns:
            - self        

        Examples:
            >>> spe_ot = speciesot.normalize_otp()
        """
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
        """dashboard
        
        Function to generate dashboards

        Args:
            - target_species_pairs (str): "human_mouse” etc. (Use “Species” instead of "Species_labels")
            - gene_list (list): List of genes to be analyzed
            - n (int): For each gene you want to analyze, how many corresponding genes do you want to find?
        
        Outputs:
            - dashboards
 
        Examples:
            >>> spe_ot.dashboard(target_species_pairs, target_genes, top_n)
        """
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
        """_create_a_list_of_genes_to_be_plotted
        
        Function to creat a list of genes to be plotted

        This function is used in SpeciesOT class function "plot_corresponding_gene_expressions"   
        """
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
        """plot_corresponding_gene_expressions
        
        Function to create line graphs of gene expression levels arranged in a grid where each row corresponds to a species and the number of columns is determined by the length of gene lists.
        
        Args:
            - target_species_pairs (string): "human_mouse” etc.
            - spe_gene_dict (dict) : Gene name notation for each species.  ex. spe_gene_dict["mouse"] = "capitalized_italic", spe_gene_dict["human"] = "all_capitalized_italic"
            - target_genes (list): List of genes to be analyzed
            - top_n (int): For each gene you want to analyze, how many corresponding genes do you want to find?

        Outputs:
            - Line graphs of gene expression

        Examples:
            >>> spe_ot.plot_corresponding_gene_expressions(self, target_species_pairs, spe_gene_dict, target_genes, top_n)        
        """
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



    def all_gene_expressions_heatmap4(
        self,
        target_species_pairs,
        target_genes,
        top_n,
        width1="3%",
        height1="100%",
        space1=0.08,
        cbar_fontsize1=12,
        cbar_ticks1=[0, 2, 4, 6, 8],
        gene_fontsize1=14,
        title_fontsize1=20,
        pad1=12,
        square_option1=True,
        cell_size1 = 0.5,
        linewidths1=0.5,
        xticklabels_option1=True,
        width2="4%",
        height2="100%",
        space2=0.115,
        cbar_fontsize2=12,
        cbar_ticks2=[0, 2, 4, 6, 8],
        gene_fontsize2=14,
        title_fontsize2=20,
        pad2=12,
        square_option2=True,
        cell_size2=0.5,
        linewidths2=0.5,
        xticklabels_option2=True,
    ):
        """all_gene_expressions_heatmap4
        
        Function to creat a customizable heatmap for cbar position, length, thickness, etc.

        This function is for developer
        """
        # Split target_species_pairs
        spe1, spe2 = target_species_pairs.split("_")

        # Obtain cell name
        cells = {}
        for spe in self.species:
            cells[spe] = self.adata[spe].obs.index.to_list()

        # Create reference gene expression dataframe
        if self.data_option == "dataset1":
            dataset1_bool = True
        else:
            dataset1_bool = False
        reference_df = self._create_reference_gene_expression_dataframe(dataset1_bool)

        # Obtain the expression level of the target genes and corresponding genes
        df1_transposed, df2_transposed = \
            self._create_gene_expression_integrated_dataframe(
                target_species_pairs, target_genes, top_n, spe1, spe2, reference_df
            )

        # Unify color scale
        vmin = min(df1_transposed.min().min(), df2_transposed.min().min())
        vmax = max(df1_transposed.max().max(), df2_transposed.max().max())


        # spe1
        num_genes = df1_transposed.shape[0]
        num_cells = df1_transposed.shape[1]

        # Heatmap size setting
        if square_option1:
            fig_width1 = num_cells * cell_size1
            fig_height1 = num_genes * cell_size1
        else:
            fig_width1 = max(num_cells * 0.015, 40)
            fig_height1 = max(5, min(num_genes * 1.2, 80))

        fig1, ax1 = plt.subplots(figsize=(fig_width1, fig_height1))

        # Drawing a heat map of spe1
        sns_heatmap1 = sns.heatmap(
            df1_transposed,
            ax=ax1,
            cmap="coolwarm",
            annot=False,
            fmt=".1f",
            xticklabels=xticklabels_option1,
            linewidths=linewidths1,
            square=square_option1,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, ha="right", fontsize=gene_fontsize1)

        # Color bar setting
        cax = inset_axes(
            ax1,
            width=width1,  
            height=height1,  
            loc="right",
            bbox_to_anchor=(space1, 0.0, 1, 1),  
            bbox_transform=ax1.transAxes,
            borderpad=0,
        )

        # Insert color bar
        cbar = fig1.colorbar(sns_heatmap1.collections[0], cax=cax, ticks=cbar_ticks1)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=cbar_fontsize1)

        # Cell name label setting
        cells_fontsize1 = gene_fontsize1
        if xticklabels_option1 == True:
            ax1.set_xticklabels(cells[spe1], fontsize=cells_fontsize1)

        ax1.set_title(f"{spe1}", fontsize=title_fontsize1, pad=pad1)

        ax1.set_ylabel("")

        plt.show()


        # spe2
        num_genes = df2_transposed.shape[0]
        num_cells = df2_transposed.shape[1]

        # Heat map size setting
        if square_option2:
            fig_width2 = num_cells * cell_size2
            fig_height2 = num_genes * cell_size2
        else:
            fig_width2 = max(num_cells * 0.015, 40)
            fig_height2 = max(15, min(num_genes * 1.2, 80))

        fig2, ax2 = plt.subplots(figsize=(fig_width2, fig_height2))

        # Drawing a heatmap of spe2
        sns_heatmap2 = sns.heatmap(
            df2_transposed,
            ax=ax2,
            cmap="coolwarm",
            annot=False,
            fmt=".1f",
            xticklabels=xticklabels_option2,
            linewidths=linewidths2,
            square=square_option2,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, ha="right", fontsize=gene_fontsize2)

        # Color bar setting
        cax = inset_axes(
            ax2,
            width=width2,  
            height=height2,  
            loc="right",
            bbox_to_anchor=(space2, 0.0, 1, 1),  
            bbox_transform=ax2.transAxes,
            borderpad=0,
        )

        # Insert color bar
        cbar = fig2.colorbar(sns_heatmap2.collections[0], cax=cax, ticks=cbar_ticks2)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=cbar_fontsize2)

        # Cell name label setting
        cells_fontsize2 = gene_fontsize2
        if xticklabels_option2 == True:
            ax2.set_xticklabels(cells[spe2], fontsize=cells_fontsize2)

        ax2.set_title(f"{spe2}", fontsize=title_fontsize2, pad=pad2)

        ax2.set_ylabel("")

        plt.show()



    def _create_a_corresponding_gene_list_for_heatmap(self, target_species_pairs, target_genes, top_n):
        """_create_a_corresponding_gene_list_for_heatmap
        
        Function to creat a corresponding gene list for heatmap

        This function is used in SpeciesOT class function "corresponding_gene_expressions_heatmap"   
        """
        corresponding_genes = {}
        heatmap_genes = []

        for key in target_genes:
            corresponding_genes[key] = (
                self.dfp_ss[target_species_pairs][key]["gene"].head(top_n).to_list()
            )
            heatmap_genes += corresponding_genes[key]

        return heatmap_genes
    


    def _create_reference_gene_expression_dataframe(self, dataset1_bool=False):
        """_create_reference_gene_expression_dataframe
        
        Function to return a new Dataframe with the average of two data points from the same period when data_option is “dataset1,” 
        and returns the original Dataframe as is when it is not.

        This function is used in SpeciesOT class function "corresponding_gene_expressions_heatmap"   
        """
        reference_df = {}

        if dataset1_bool:
            # Create a new data frame averaged over two contemporaneous data points
            for key in self.species:
                reference_df[key] = _merge_pairs_with_average(self.plot_normalized_log_select_preprocessed_masked[key])
        else:
            for key in self.species:
                reference_df[key] = self.plot_normalized_log_select_preprocessed_masked[key]

        return reference_df



    def _create_gene_expression_integrated_dataframe(
            self, target_species_pairs, target_genes, top_n, spe1, spe2, reference_df
    ):    
        """_create_gene_expression_integrated_dataframe
        
        Function to creat a Dataframe of the expression levels of the target genes and the corresponding genes within nth for given genes

        This function is used in SpeciesOT class function "corresponding_gene_expressions_heatmap"   
        """
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
        """_plot_heatmap
        
        Function to plot heatmap

        This function is used in SpeciesOT class function 
        "corresponding_gene_expressions_heatmap" and "corresponding_gene_expressions_separated_heatmap"
        """
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
        """_plot_heatmap
        
        Function to plot heatmap and color bar separately

        This function is used in SpeciesOT class function "corresponding_gene_expressions_separated_heatmap"   
        """
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



    def corresponding_gene_expressions_heatmap(
        self, target_species_pairs, spe_gene_dict, target_genes, top_n, dataset1_bool=False, raw_return_opt=False
    ):
        """corresponding_gene_expressions_heatmap
        
        Function to creat a heatmap of the expression levels of the target genes and the corresponding genes within nth for given genes

        Args:
            - target_species_pairs (string): "human_mouse” etc.
            - spe_gene_dict (dict) : Gene name notation for each species.  ex. spe_gene_dict["mouse"] = "capitalized_italic", spe_gene_dict["mouse"] = "all_capitalized_italic"
            - target_genes (list): List of genes to be analyzed
            - top_n (int): For each gene you want to analyze, how many corresponding genes do you want to find?
            - dataset1_bool (bool) : Whether to correct the data to full time series data before plotting
            - raw_return_opt (bool): Whether to return raw data
        
        Outputs:
            - Gene expression heatmap of spe1
            - Gene expression heatmap of spe2
 
        Examples:
            >>> spe_ot.corresponding_gene_expressions_heatmap(target_species_pairs, spe_gene_dict, target_genes, top_n)
        """
        # Split target_species_pairs
        spe1, spe2 = target_species_pairs.split("_")

        # Obtain cell name
        cells = {}
        for spe in self.species:
            cells[spe] = self.adata[spe].obs.index.to_list()

        # Create reference gene expression dataframe
        reference_df = self._create_reference_gene_expression_dataframe(dataset1_bool)

        # Obtain the expression level of the target genes and corresponding genes
        integrated_df1_transposed, integrated_df2_transposed = \
            self._create_gene_expression_integrated_dataframe(
                target_species_pairs, target_genes, top_n, spe1, spe2, reference_df
            )
 
        # Unify color scale
        vmin = min(integrated_df1_transposed.min().min(), integrated_df2_transposed.min().min())
        vmax = max(integrated_df1_transposed.max().max(), integrated_df2_transposed.max().max())

        # Plot heatmap of spe1
        self._plot_heatmap(spe1, cells, spe_gene_dict, dataset1_bool, integrated_df1_transposed, vmin, vmax)

        if raw_return_opt:
            print("The returned object (`integrated_df1_transposed`) is pd.DataFrame")

        # Plot heatmap of spe2
        self._plot_heatmap(spe2, cells, spe_gene_dict, dataset1_bool, integrated_df2_transposed, vmin, vmax)

        if raw_return_opt:
            print("The returned object (`integrated_df2_transposed`) is pd.DataFrame")

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print("The number of objects that can be returned is 2")
            return integrated_df1_transposed, integrated_df2_transposed



    def corresponding_gene_expressions_separated_heatmap(
        self, target_species_pairs, spe_gene_dict, target_genes, top_n, dataset1_bool=False, cbar_separate_opt=False, raw_return_opt=False
    ):
        """corresponding_gene_expressions_separated_heatmap
        
        Function to creat a heatmap of the expression levels of the target genes and the corresponding genes within nth for given genes

        Args:
            - target_species_pairs (string): "human_mouse” etc.
            - spe_gene_dict (dict) : Gene name notation for each species.  ex. spe_gene_dict["mouse"] = "capitalized_italic", spe_gene_dict["human"] = "all_capitalized_italic"
            - target_genes (list): List of genes to be analyzed
            - top_n (int): For each gene you want to analyze, how many corresponding genes do you want to find?
            - dataset1_bool (bool) : Whether to correct the data to full time series data before plotting
            - raw_return_opt (bool): Whether to return raw data
        
        Outputs:
            - Gene expression heatmap of spe1
            - Gene expression heatmap of spe2
 
        Examples:
            >>> spe_ot.corresponding_gene_expressions_heatmap(target_species_pairs, spe_gene_dict, target_genes, top_n)
        """
        # Split target_species_pairs
        spe1, spe2 = target_species_pairs.split("_")

        # Obtain cell name
        cells = {}
        for spe in self.species:
            cells[spe] = self.adata[spe].obs.index.to_list()
    
        # Create reference gene expression dataframe
        reference_df = self._create_reference_gene_expression_dataframe(dataset1_bool)

        # Obtain the expression level of the target genes and corresponding genes
        integrated_df1_transposed, integrated_df2_transposed = \
            self._create_gene_expression_integrated_dataframe(
                target_species_pairs, target_genes, top_n, spe1, spe2, reference_df
            )
        
        # Unify color scale
        vmin = min(integrated_df1_transposed.min().min(), integrated_df2_transposed.min().min())
        vmax = max(integrated_df1_transposed.max().max(), integrated_df2_transposed.max().max())


        # Plot a heat map for each target gene
        for target_gene in target_genes:
            # Obtain the expression level of the target genes
            df1 = reference_df[spe1][target_gene]

            # df1 is a Series, so convert it to a data frame and set the column name to target_gene.
            df1_transposed = pd.DataFrame(df1).T
            df1_transposed.index.name = "gene"

            # Obtain corresponding genes
            heatmap_genes = self.dfp_ss[target_species_pairs][target_gene]["gene"].head(top_n).to_list()

            # Obtain the expression level of the corresponding genes
            df2 = reference_df[spe2][heatmap_genes]
            df2_transposed = df2.T

            if cbar_separate_opt:
                # Only display color bar for the last heatmap 
                if target_gene == target_genes[-1]:
                    cbar_opt = True
                else:
                    cbar_opt = False

                # Plot haatmap and color bar
                self._plot_heatmap_and_colorbar_separately(spe, cells, spe_gene_dict, dataset1_bool, df1_transposed, vmin, vmax, cbar_opt=False)
                self._plot_heatmap_and_colorbar_separately(spe, cells, spe_gene_dict, dataset1_bool, df2_transposed, vmin, vmax, cbar_opt=cbar_opt)
            else:
                # Plot heatmap of spe1
                self._plot_heatmap(spe1, cells, spe_gene_dict, dataset1_bool, df1_transposed, vmin, vmax)

                # Plot heatmap of spe2
                self._plot_heatmap(spe2, cells, spe_gene_dict, dataset1_bool, df2_transposed, vmin, vmax)

            # Return raw data
            if raw_return_opt:
                print("# Raw data summary")
                print("The number of objects that can be returned is 2")
                print("Both of them are pd.DataFrame")
                return df1_transposed, df2_transposed



    def _generate_histogram_of_values_of_normalized_otp(self, data, pair):
        """_generate_histogram_of_values_of_normalized_otp
        
        Function to generate a Histogram of the values of normalized OTP

        Outputs:
            - Histogram of values of NOTP

        This is used in SpeciesOT class function "distribution_of_values"      
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(data, log=True)
        ax.set_xlabel("Values")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title(f"Histogram of values of NOTP ( {self.data_option},  {pair})")
        plt.show()



    def _sort_df_by_max_notp_value_for_each_col(self, pair):
        """_sort_df_by_max_notp_value_for_each_col
        
        Function to sort DataFrame based on the maximum notp value for each col

        This is used in SpeciesOT function "distribution_of_values"
        """
        # Step 1: Compute the maximum value for each column
        df = self.df_coupling_matrices_normalized[pair]
        max_values = df.max()

        # Step 2: Sort the column names based on their maximum values
        sorted_columns = max_values.sort_values(ascending=False).index
        sorted_columns = max_values.sort_values(ascending=True).index

        # Step 3: Reorder the DataFrame columns based on sorted max values
        sorted_df = df[sorted_columns]   

        return sorted_df     



    def _generate_box_plots_of_values_in_each_col(self, sorted_df):
        """_generate_box_plots_of_values_in_each_col
        
        Function to generate Box plots of values in each col (asc)

        Outputs:
            - Box plots of values in each col (asc)

        This is used in SpeciesOT class function "distribution_of_values"      vunni7-fehjeh-wEhxi
        """
        # Now you can create a boxplot for the sorted DataFrame
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_and_selected_df = sorted_df.iloc[:, 0:10]
        sorted_and_selected_df.boxplot(ax=ax)

        # Set labels and title
        ax.set_title("Box plots of values in each col (asc)")
        ax.set_xlabel("Col")
        ax.set_ylabel("Values")

        plt.show()



    def _calc_transposed_notp(self, pair):
        """_calc_transposed_notp
        
        Function to calculate transposed notp
        """
        # Transpose the dataframe so that rows become columns
        df = self.df_coupling_matrices_normalized[pair]
        transposed_notp = df.T

        return transposed_notp



    def _generate_box_plots_of_values_in_each_row(self, transposed_notp):
        """_generate_box_plots_of_values_in_each_row
        
        Function to generate Box plots of values in each row

        Outputs:
            - Box plots of values in each col (asc)

        This is used in SpeciesOT class function "distribution_of_values"      
        """
        # Create box plots for each row (which are now columns after transposition)
        fig, ax = plt.subplots(figsize=(10, 6))
        transposed_notp.boxplot(ax=ax)

        # Set labels and title
        ax.set_title("Box plots of values in each row")
        ax.set_xlabel("Row")
        ax.set_ylabel("Values")

        plt.show()



    def distribution_of_values(self, pair="", raw_return_opt=False):
        """distribution_of_values
        
        Function to check the distribution of the values 
        of the normalized optimal transport plans

        Args:
            - pair (str): Which species pair's NOTP to look at
            - raw_return_opt (bool): Whether to return raw data 

        Outputs:
            - Histogram of values of NOTP
            - Box plots of values in each col (asc)
            - Box plots of values in each row

        Examples:
            >>> spe_ot.distribution_of_values()
        """
        # Set species_pair
        if pair == "":
            pair = self.species_pairs[1]


        # Histogram of values of NOTP
        values_of_notp = self.df_coupling_matrices_normalized[pair].values.flatten()
        self._generate_histogram_of_values_of_normalized_otp(values_of_notp, pair)

        if raw_return_opt:
            print("The returned object (`values_of_notp`) is np.darray")


        # Box plots of values in each col (asc)
        sorted_df = self._sort_df_by_max_notp_value_for_each_col(pair)
        self._generate_box_plots_of_values_in_each_col(sorted_df)

        if raw_return_opt:
            print("The returned object (`sorted_df`) is pd.DataFrame")
            print("To be precise, `sorted_df[:, 0:10]` is being plotted as a box plot")


        # Box plots of values in each rows
        transposed_notp = self._calc_transposed_notp(pair)
        self._generate_box_plots_of_values_in_each_row(transposed_notp)

        if raw_return_opt:
            print("The returned object (`transposed_notp`) is pd.DataFrame")


        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print("The number of objects that can be returned is 3")
            return values_of_notp, sorted_df, transposed_notp



    def _get_ranked_genes_per_gene_by_normalized_otp(self):
        """_get_ranked_genes_per_gene_by_normalized_otp
        
        Function to sort the normalized otp columns by row size to obtain a list of ranked genes by each gene

        This is used in SpeciesOT class function "ccp"      
        """
        value = 0.0
        sorted_column_names_df = {}

        for key in self.species_pairs:
            df = self.df_coupling_matrices_normalized[key].where(
                self.df_coupling_matrices_normalized[key] >= value, np.nan
            )

            # Get the sorted column names dataframe with NaN positions considered
            sorted_column_names_df[key] = pd.DataFrame(
                df.apply(
                    lambda row: row.sort_values(ascending=False).index.tolist(), axis=1
                ).tolist(),
                index=df.index,
            )

            # Exclude genes for which no data exist in df from the ranked list
            for i, row in df.iterrows():
                sorted_column_names_df[key].loc[i] = [
                    col if not pd.isna(row[col]) else np.nan
                    for col in sorted_column_names_df[key].loc[i]
                ]

        return sorted_column_names_df
    


    def _calculate_top_n_gene_length_and_coverage(self, sorted_column_names_df):
        """_calculate_top_n_gene_length_and_coverage
        
        Function to calculate top n gene length and coverage 

        This is used in SpeciesOT class function "ccp"      
        """
        topN_end = 1 + min(df.shape[0] for df in self.df_coupling_matrices.values())

        unique_strings = {}
        len_unique_strings = {}
        percent_unique_strings = {}

        for key in self.species_pairs:
            unique_strings[key] = {}
            len_unique_strings[key] = {}
            percent_unique_strings[key] = {}

            for n in range(1, topN_end):
                # Get the set of unique strings appearing in the first n columns of sorted_column_names_df
                unique_strings[key][n] = set(
                    sorted_column_names_df[key].iloc[:, :n].to_numpy().ravel()
                )
                unique_strings[key][n].discard(np.nan)
                len_unique_strings[key][n] = len(unique_strings[key][n])
                percent_unique_strings[key][n] = (
                    len_unique_strings[key][n] 
                    / self.df_coupling_matrices[key].shape[1]  # Use all
                )

        return len_unique_strings, percent_unique_strings        
    


    def _create_initial_ccp_dataframe(self, species_pair, len_unique_strings, percent_unique_strings):
        """_create_initial_ccp_dataframe
        
        Function to create an initial ccp dataframe of the specified species pairs

        This is used in SpeciesOT class function "ccp"
        """
        # Extract len_unique_strings and percent_unique_strings for the current species pair
        n_values = list(len_unique_strings[species_pair].keys())
        len_values = list(len_unique_strings[species_pair].values())
        percent_values = list(percent_unique_strings[species_pair].values())

        ccp_ = pd.DataFrame(
            {
                "count": len_values,
                "all": self.df_coupling_matrices[species_pair].shape[1],
                "CCP": percent_values,
            },
            index=n_values,
        )       

        return ccp_ 



    def ccp(self):
        """ccp
        
        Function to create cumulative coverage percent DataFrame

        Outputs:
            >>> self.ccp()
        """
        # Sort columns of normalized otp by magnitude per row
        sorted_column_names_df = self._get_ranked_genes_per_gene_by_normalized_otp()

        # Get top n gene set and calculate its length and proporation
        len_unique_strings, percent_unique_strings = self._calculate_top_n_gene_length_and_coverage(sorted_column_names_df)
        self.percent_unique_strings = percent_unique_strings

        ccp_ = {}
        self.cumulative_coverage_percentage = {}

        # Step 1: Create initial CCP DataFrames 
        for key in self.species_pairs:
            ccp_[key] = self._create_initial_ccp_dataframe(key, len_unique_strings, percent_unique_strings)


        # Step 2: Calculate CCP using ccp_[key] and ccp_[reversed_key]
        for key in self.species_pairs:
            # Define the reversed key by swapping the species in the pair
            spe1, spe2 = key.split("_")
            reversed_key = f"{spe2}_{spe1}"

            if reversed_key in ccp_:
                # Compare the "all" values to determine which DataFrame to use
                if ccp_[key]["all"].iloc[0] >= ccp_[reversed_key]["all"].iloc[0]:
                    self.cumulative_coverage_percentage[key] = ccp_[key].copy()
                else:
                    self.cumulative_coverage_percentage[key] = ccp_[reversed_key].copy()
            else:
                # If the reversed key doesn't exist, use the original data
                self.cumulative_coverage_percentage[key] = ccp_[key].copy()
               

        return self



    def _generate_cumulative_coverage_count(self):
        """_generate_cumulative_coverage_count
        
        Function to generate dendrogram derived from sinkhorn entropy gw distance

        This is used in SpeciesOT class function "plot_ccp"   
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Iterate over Species_pairs with a specific step size to match the plotting strategy
        for i, key in enumerate(self.species_pairs[:: len(self.species)]):
            x_values = self.cumulative_coverage_percentage[key].index.tolist()  # n values (1, ..., topN_end-1)
            y_values = self.cumulative_coverage_percentage[key]["count"].tolist()  # "count" column from CCP[key]

            # Select marker styles for different keys
            markers = ["o", "v", "^", "<", ">", "x"]
            marker = markers[i % len(markers)]

            plt.plot(x_values, y_values, marker=marker, label=key)

        ax.set_title("Cumulative Coverage Count")
        ax.set_xlabel("n (Number of Columns)")
        ax.set_ylabel("Unique Strings Count")
        ax.legend()
        plt.grid(True)
        plt.show()



    def _generate_cumulative_ratios_by_optimal_transport_plans(self):
        """_generate_cumulative_ratios_by_optimal_transport_plans
        
        Function to generate Cumulative ratios determined by optimal transport plans

        This is used in SpeciesOT class function "plot_ccp", "outputs_for_paper"        
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Iterate over Species_pairs with a specific step size to match the plotting strategy
        for i, key in enumerate(self.species_pairs[:: len(self.species)]):
            x_values = self.cumulative_coverage_percentage[key].index.tolist()  # n values (1, ..., topN_end-1)
            y_values = self.cumulative_coverage_percentage[key]["CCP"].tolist()  # "CCP" column from CCP[key]

            # Select marker styles for different keys
            markers = ["o", "v", "^", "<", ">", "x"]
            marker = markers[i % len(markers)]

            plt.plot(x_values, y_values, marker=marker, label=key)

        ax.set_title("Cumulative ratios determined by optimal transport plans")
        ax.set_xlabel("Rank r")
        ax.set_ylabel("Cumative ratio [%]")
        ax.set_xlim(0, 40)

        ax.legend()
        plt.grid(True)
        plt.show()



    def _print_discription_of_self_dot_ccp(self):
        """_print_discription_of_self_dot_ccp
        
        Function to print discription of `self.cumulative_coverage_percentage`

        This is used in SpeciesOT class function "plot_ccp", "outputs_for_paper"  
        """
        print("The returned objects (`self.cumulative_coverage_percentage`) is a dict")
        print("The key of `self.cumulative_coverage_percentage` is species_pair")
        print("The value of `self.cumulative_coverage_percentage` is pd.DataFrame")
        print("The column names in this DataFrame are “count”, “all” and “CCP” ")



    def plot_ccp(self, raw_return_opt=False):
        """plot_ccp
        
        Function to plot cumurateve coverage count and cumulative ratios determined by optimal transport plans

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Outputs:
            - Cumulative Coverage Count
            - Cumulative ratios determined by optimal transport plans

        Examples:
            >>> spe_ot.plot_ccp()
        """
        # Cumulative Coverage Count
        self._generate_cumulative_coverage_count()

        # Cumulative ratios determined by optimal transport plans
        self._generate_cumulative_ratios_by_optimal_transport_plans()

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print("The number of objects that can be returned is 1")
            self._print_discription_of_self_dot_ccp()
            return self.cumulative_coverage_percentage



    def _calculate_l2_distance_between_speciies_pairs(self):
        """_calculate_l2_distance_between_l2_distance_speciies_pairs
        
        Functioin to calculate L2 distance between species_pairs

        This is used in SpeciesOT class function "l2_distance_of_ccp_plots"
        """
        l2_distances = {}
        keys = self.species_pairs

        for i in range(len(keys)):
            for j in range(
                i + 1, len(keys)
            ):  # Avoid redundant comparisons and self-comparisons
                key1 = keys[i]
                key2 = keys[j]

                # Extract the underlying NumPy array
                df1 = self.cumulative_coverage_percentage[key1]["CCP"].to_numpy() 
                df2 = self.cumulative_coverage_percentage[key2]["CCP"].to_numpy() 

                # Calculate the element-wise squared differences
                squared_diff = (df1 - df2) ** 2

                # Calculate the L2 distance by summing the squared differences and taking the square root
                l2_distance = np.sqrt(squared_diff.sum())

                # Store the L2 distance in a dictionary for the key pair
                l2_distances[(key1, key2)] = l2_distance

        return l2_distances

    

    
    def l2_distance_of_ccp_plots(self):
        """l2_distance_of_ccp_plots
        
        Function to calculate species difference from CCP plots using L2 distance

        Returns:
            - self

        Examples:
            >>> spe_ot = spe_ot.l2_distance_of_ccp_plots()
        """
        # Calculate L2 distance between species_pairs
        l2_distances = self._calculate_l2_distance_between_speciies_pairs()

        # Create distance matrix between species
        self.distance_matrix_ccp = pd.DataFrame(
            index=self.species, columns=self.species, dtype=float
        )  # Specify dtype as float

        for (pair1, pair2), distance in l2_distances.items():
            # Extract species indices from pairs
            pair1_species_0 = pair1.split("_")[0]
            pair1_species_1 = pair1.split("_")[1]
            pair2_species_0 = pair2.split("_")[0]
            pair2_species_1 = pair2.split("_")[1]

            for idx, species in enumerate(self.species):
                # Case 1: `species_species` vs `species_other`
                if (
                    pair1_species_0 == species
                    and pair1_species_1 == species
                    and pair2_species_0 == species
                ):
                    species1 = species
                    species2 = pair2_species_1
                    self.distance_matrix_ccp.loc[species1, species2] = distance

                # Case 2: `species_species` vs `other_species`
                elif (
                    pair1_species_0 == species
                    and pair1_species_1 == species
                    and pair2_species_1 == species
                ):
                    species1 = species
                    species2 = pair2_species_0
                    self.distance_matrix_ccp.loc[species2, species1] = distance


        # Set the diagonal to 0 (self-distance is always 0)
        for species in self.species:
            self.distance_matrix_ccp.loc[species, species] = 0.0

        return self
    


    def cumulative_ratios_dismat(self, raw_return_opt=False):
        """cumulative_ratios_dismat
        
        Function to plot distance matrix derived from cumulative ratios

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Examples:
            >>> spe_ot.cumulative_ratios_dismat()
        """
        ax = sns.heatmap(self.distance_matrix_ccp, annot=True, fmt=".3f")
        if self.data_option == "dataset2":
            ax.set_xticklabels(self.species_labels, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(self.species_labels, rotation=0)
        ax.set_yticklabels(self.species_labels, rotation=0)
        plt.title("Distance matrix derived from cumulative ratios")
        plt.show()

        # Return raw data
        if raw_return_opt:
            print("# Raw data summary")
            print(f"The number of objects that can be returned is 1")
            print("The returned object (self.distance_matrix_ccp`) is a linkage matrix (np.darray)")
            return self.distance_matrix_ccp



    def _set_dismat_and_labels_for_ccp_dendrogram(self):
        """_set_dismat_and_labels_for_ccp_dendrogram
        
        Function to set a distance matirix and labels for ccp dendrogram

        This is used in SpeciesOT class function "ccp_dendrogram" and "outputs_for_paper"
        """
        custom_color = "#ff0000"

        # Reverse the input distance matrix both row-wise and column-wise
        reversed_distance_matrix_ccp = np.flipud(np.fliplr(self.distance_matrix_ccp))

        # Perform hierarchical clustering on the reversed matrix
        linked_ccp = linkage(pdist(reversed_distance_matrix_ccp), "ward")

        # Define your labels
        labels = self.species_labels
        # Reversed labels to match the reversed matrix
        labels = labels[::-1]

        return custom_color, linked_ccp, labels
    


    def _plot_ccp_dendrogram(
            self, custom_color, linked_ccp, labels, title_fontsize=14, xlabel_fontsize=14
        ):
        """_set_dismat_and_labels_for_ccp_dendrogram
        
        Function to plot ccp dendrogram

        This is used in SpeciesOT class function "ccp_dendrogram" and "outputs_for_paper"
        """
        fig, ax = plt.subplots(figsize=(6, 3))

        dendrogram(
            linked_ccp,
            orientation="left",
            labels=labels,  
            distance_sort="descending",
            show_leaf_counts=True,
            color_threshold=np.inf,  # Set to infinity to apply the color to all links
            link_color_func=lambda k: custom_color,  # Apply custom color
        )

        ax.invert_yaxis()  # Invert the y-axis to reverse the dendrogram vertically
        ax.set_title("Dendrogram derived from cumulative ratios", fontsize=title_fontsize)
        ax.set_xlabel("$\\ell^2$ similarity", fontsize=xlabel_fontsize)
        
        plt.show()        



    def ccp_dendrogram(self, raw_return_opt=False):
        """ccp_dendrogram
        
        FUnction to plot dendrogram derived from cumulative ratios

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Outputs:
            - Dendrogram derivedd from cumulative ratios
        
        Examples:
            >>> spe_ot.ccp_dendrogram()
        """
        custom_color, linked_ccp, labels = self._set_dismat_and_labels_for_ccp_dendrogram()

        self._plot_ccp_dendrogram(
            custom_color, linked_ccp, labels, title_fontsize=14, xlabel_fontsize=14
        )

        # Return raw data
        if raw_return_opt:
            print(f"The number of objects that can be returned is 1")
            print("The returned object (`linked_ccp`) is a linkage matrix (np.darray)")
            return linked_ccp



    def outputs_for_paper(self, raw_return_opt=False):
        """outoputs_for_paper
        
        Function to output all major results at once

        Args:
            - raw_return_opt (bool): Whether to return raw data

        Outputs:
            - Entropy GW distance
            - Sinkhorn entropy GW distance
            - Transcriptomic discrepancy tree
            - Cumulative ratios determined by optimal Stransport plans
            - Distance matrix derived from cumulative ratios
            - Dendrogram derived from cumulative ratios

        Examples:
            >> spe_ot.outputs_for_paper()
        """
        # List of objects that can be returned
        returnable_object_type_list = []


        # Entropy GW distance
        entropy_gw_distance = self._calculate_entropy_gw_distance()
        self._visualize_entropy_gw_distance(entropy_gw_distance)

        returnable_object_type_list = _append_to_returnable_object_type_list(
            raw_return_opt, returnable_object_type_list, "heatmap", "entropy_gw_distance"
        )


        # Sinkhorn entropy GW distance
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


        # Cumulative ratios determined by optimal transport plans
        self._generate_cumulative_ratios_by_optimal_transport_plans()

        if raw_return_opt:
            self._print_discription_of_self_dot_ccp()
            returnable_object_type_list.append("dict")


        # Distance matrix derived from cumulative ratios
        self.cumulative_ratios_dismat()

        returnable_object_type_list = _append_to_returnable_object_type_list(
            raw_return_opt, returnable_object_type_list, "heatmap", "self.distance_matrix_ccp"
        )


        # Dendrogram derived from cumulative ratios
        custom_color, linked_ccp, labels = self._set_dismat_and_labels_for_ccp_dendrogram()
        self._plot_ccp_dendrogram(
            custom_color, linked_ccp, labels, title_fontsize=14, xlabel_fontsize=14
        )

        returnable_object_type_list = _append_to_returnable_object_type_list(
            raw_return_opt, returnable_object_type_list, "dendrogram", "linked_ccp"
        )


        # Return raw data
        if raw_return_opt:
            print(f"The number of objects that can be returned is {len(returnable_object_type_list)}")
            return entropy_gw_distance, sinkhorn_entropy_gw_distance, \
                ordered_linkage, self.cumulative_coverage_percentage, \
                    self.distance_matrix_ccp, linked_ccp

