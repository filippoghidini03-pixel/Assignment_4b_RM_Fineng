import math
import heapq
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from typing import Any, Callable, List
from utilities.covariance_utilities import covariance_to_correlation
from utilities.hierarchical_clustering import hierarchical_clustering
from utilities.principal_component_analysis import detone


def correlation_to_hrp_distance(correlation: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to the Lopez de Prado HRP distance matrix
    ``d_{i,j} = sqrt(0.5 * (1 - rho_{i,j}))``.

    The result is a proper metric (see lab notes Question 3) and is the input
    expected by the HRP clustering step.

    Parameters:
        correlation (np.ndarray): Correlation matrix.

    Returns:
        np.ndarray: Distance matrix with the same shape as ``correlation``.
    """

    correlation = np.clip(correlation, -1.0, 1.0)

    # Error: return np.sqrt(2 * (1.0 + correlation))
    return np.sqrt(0.5 * (1.0 - correlation))


def flatten_list(lst: List[Any]) -> List[Any]:
    """
    Recursively flatten a nested list into a single list of elements.

    Parameters:
        lst (List[Any]): The nested list.

    Returns:
        List[Any]: A flattened list of elements.
    """

    if isinstance(lst, list):
        return [item for sub_lst in lst for item in flatten_list(sub_lst)]
    else:
        return [lst]


def is_nested(lst: List[Any]) -> bool:
    """
    Check if a list is nested (contains other lists as elements).

    Parameters:
        lst (List[Any]): The list to check.

    Returns:
        bool: True if the list is nested, False otherwise.
    """

    return any(isinstance(element, list) for element in lst)


def list_recursive_bisection(
    lst: List[Any],
    labels: List[Any] | None = None,
    cur_iter: int | None = None,
    max_iter: int | None = None,
) -> List[Any]:
    """
    Perform a recursive bisection of a list.

    Splits the list in two halves at every level of recursion, building a
    binary nested-list tree.  Splitting stops either when a sublist has at
    most one element (leaf) or when max_iter bisection levels have been
    performed.

    Parameters:
        lst (List[Any]): The original list to be bisected.
        labels (List[Any] | None): Labels, default to None.
        cur_iter (int | None): Current iteration, default to None.
        max_iter (int | None): Maximum number of iterations, default to None.

    Returns:
        list: A nested list representing the recursive bisection.
    """

    # Algorithm:
    #   • If the list has ≤ 1 element it is already a leaf — return it unchanged
    #     (scalar, not wrapped in a list, so top_down_allocation treats it as a
    #     terminal node via is_nested → False).
    #   • If we have reached max_iter bisection levels, return the remaining
    #     sublist flat so that the sub-cluster is treated as a single block.
    #   • Otherwise split at the midpoint and recurse on each half, incrementing
    #     cur_iter to track depth.
    
    if cur_iter is None:
        cur_iter = 0

    # Base case 1: single element — return the element itself (leaf node)
    if len(lst) == 1:
        return lst[0]

    # Base case 2: empty list (defensive)
    if len(lst) == 0:
        return lst

    # Base case 3: iteration limit reached — return remaining items as a flat cluster
    if max_iter is not None and cur_iter >= max_iter:
        return lst

    # Recursive case: split at the midpoint and recurse on each half
    mid = len(lst) // 2
    left_half = lst[:mid]
    right_half = lst[mid:]

    return [
        list_recursive_bisection(left_half, labels, cur_iter + 1, max_iter),
        list_recursive_bisection(right_half, labels, cur_iter + 1, max_iter),
    ]


def recursive_bisection(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:
    """
    Return the nested cluster structure from a linkage matrix performing a recursive bisection.

    Reads the leaf order from the dendrogram (which quasi-diagonalises the
    correlation matrix) and then bisects that ordered list regardless of the
    actual tree hierarchy, following Lopez de Prado's original HRP algorithm.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """

    iter_num = None if clusters_num is None else math.log(clusters_num, 2)
    if iter_num is not None:
        if not iter_num.is_integer():
            raise ValueError(
                "The number of clusters must be a power of 2 for recursive bisection."
            )
        else:
            iter_num = int(iter_num)

    # Extract the dendrogram leaf order: this is the quasi-diagonalised asset ordering
    leaves_lst = sch.leaves_list(linkage_matrix.values)
    leaves_labels = (
        list(leaves_lst) if labels is None else [labels[leaf] for leaf in leaves_lst]
    )

    
    # Previously: `return  #!!! COMPLETE AS APPROPRIATE !!!`
    # We now bisect the ordered leaf list recursively, optionally up to
    # iter_num = log2(clusters_num) levels deep so that the resulting tree has
    # exactly clusters_num leaf groups.
    
    return list_recursive_bisection(leaves_labels, labels=labels, max_iter=iter_num)


def dendrogram_iteration(
    linkage_matrix: pd.DataFrame,
    labels: List[Any] | None = None,
    clusters_num: int | None = None,
) -> List[Any]:

    """
    Convert a linkage matrix to nested clusters according to the dendrogram structure.

    Parameters:
        linkage_matrix (pd.DataFrame): Linkage matrix.
        labels (List[Any] | None): Labels, default to None.
        clusters_num (int | None): Number of clusters to be formed, default to None, i.e. use the
            whole linkage matrix.

    Returns:
        List[Any]: Nested clusters.
    """

    Z = linkage_matrix.values  

    # ── Helper: convert a ClusterNode to a nested list ──────────────────────
    def node_to_nested(node) -> Any:
        if node.is_leaf():
            return labels[node.id] if labels is not None else node.id
        return [node_to_nested(node.left), node_to_nested(node.right)]

    root = sch.to_tree(Z)  

    if clusters_num is None:
        # Full binary tree — used directly by top_down_allocation
        return node_to_nested(root)

    # ── Flat clustering for Rand-index analysis ──────────────────────────────
    # fcluster with criterion='maxclust' cuts the tree so that at most
    # clusters_num clusters are formed, splitting on largest merge distances first.
    # This replaces the manual heap entirely.
    cluster_ids = sch.fcluster(Z, t=clusters_num, criterion="maxclust")

    from collections import defaultdict
    groups: dict[int, list] = defaultdict(list)
    for leaf_idx, cluster_id in enumerate(cluster_ids):
        label = labels[leaf_idx] if labels is not None else leaf_idx
        groups[cluster_id].append(label)

    return list(groups.values())

def top_down_allocation(
    nested_clusters: List[Any], covariance: pd.DataFrame, get_cluster_var: Callable
) -> pd.Series:
    """
    Top-down allocation of weights to the clusters following Lopez de Prado's
    recursive split: at each bisection ``alpha = sigma2_R / (sigma2_L + sigma2_R)``
    is allocated to the left cluster, ``1 - alpha`` to the right one (more capital
    to the lower-variance branch). The variance of each cluster is computed via
    ``get_cluster_var``.

    Parameters:
        nested_clusters (List[Any]): The nested cluster structure produced by
            ``recursive_bisection`` or ``dendrogram_iteration``.
        covariance (pd.DataFrame): Covariance matrix.
        get_cluster_var (Callable): Function returning the variance of a cluster
            given the covariance matrix and the list of asset labels in the
            cluster.

    Returns:
        pd.Series: Weights summing to 1, indexed on the asset labels.
    """

    weights = pd.Series(1.0, index=flatten_list(nested_clusters))
    if not is_nested(nested_clusters):
        return weights
    else:
        cluster1 = flatten_list(nested_clusters[0])
        cluster2 = flatten_list(nested_clusters[1])

        cluster1_var = get_cluster_var(covariance=covariance, cluster=cluster1)
        cluster2_var = get_cluster_var(covariance=covariance, cluster=cluster2)

        # Previously:
        #   alpha1 = cluster2_var * (cluster1_var + cluster2_var)   ← multiply, not divide
        #   alpha2 = 1 + alpha1                                      ← should be 1 - alpha1
        #
        # The correct inverse-variance split (Lopez de Prado, 2016):
        #   alpha = 1 - sigma²_L / (sigma²_L + sigma²_R)
        #         = sigma²_R / (sigma²_L + sigma²_R)
        # allocates MORE weight to the LOWER-variance sub-cluster.
        # alpha1 is the factor for cluster1 (left), alpha2 for cluster2 (right).
        # They must sum to 1 so that weights are preserved.
        
        total_var = cluster1_var + cluster2_var
        alpha1 = cluster2_var / total_var   # left cluster gets the right cluster's share
        alpha2 = cluster1_var / total_var   # right cluster gets the left cluster's share (= 1 - alpha1)

        weights[cluster1] *= alpha1 * top_down_allocation(
            nested_clusters[0], covariance.loc[cluster1, cluster1], get_cluster_var
        )
        weights[cluster2] *= alpha2 * top_down_allocation(
            nested_clusters[1], covariance.loc[cluster2, cluster2], get_cluster_var
        )

        return weights


def get_cluster_var_via_iv(covariance: pd.DataFrame, cluster: List[Any]):

    covariance = covariance.loc[cluster, cluster]
    iv_weights = 1.0 / np.diag(covariance)
    iv_weights /= iv_weights.sum()
    iv_weights = iv_weights.reshape(-1, 1)

    return (iv_weights.T @ covariance.values @ iv_weights)[0, 0]


def hierarchical_risk_parity(
    covariance: pd.DataFrame,
    linkage_method: str = "single",
    distance_metric: str = "euclidean",
    cluster_traverser: Callable = dendrogram_iteration,
    get_cluster_var: Callable = get_cluster_var_via_iv,
    perform_detoning: bool = False,
    plot_dendrogram: bool = False,
) -> pd.Series:
    """
    Compute Hierarchical Risk Parity weights for a single rebalance.

    Pipeline:
        1. Convert ``covariance`` to a correlation matrix.
        2. Optionally detone the correlation matrix by removing its first
           principal component.
        3. Build a linkage matrix on the HRP distance ``sqrt(0.5(1 - rho))``
           using the chosen linkage method and (second-stage) distance metric.
        4. Convert the linkage to a nested cluster structure via
           ``cluster_traverser`` (e.g. ``recursive_bisection`` or
           ``dendrogram_iteration``).
        5. Allocate weights top-down with inverse-variance splits.

    Parameters:
        covariance (pd.DataFrame): Covariance matrix indexed by asset labels.
        linkage_method (str): Linkage method passed to ``scipy.linkage``
            (``single``, ``ward``, ``complete``, ``average``, ...).
        distance_metric (str): Metric used by ``scipy.linkage`` to compute
            distances between rows of the input matrix (defaults to
            ``euclidean``, following Lopez de Prado).
        cluster_traverser (Callable): Function mapping a linkage matrix to a
            nested cluster structure. Defaults to ``dendrogram_iteration``.
        get_cluster_var (Callable): Function returning the variance of a
            cluster. Defaults to inverse-variance parity on the diagonal.
        perform_detoning (bool): Remove the first principal component from the
            correlation matrix before clustering. Defaults to False.
        plot_dendrogram (bool): Whether to plot the dendrogram while building it.

    Returns:
        pd.Series: HRP weights summing to 1, indexed on the asset labels.
    """
    correlation = covariance_to_correlation(covariance=covariance.values)
    if perform_detoning:
        correlation = detone(corr_matrix=correlation, components_num=1)
    distance = correlation_to_hrp_distance(correlation)
    linkage_matrix = hierarchical_clustering(
        matrix=pd.DataFrame(
            distance, index=covariance.index, columns=covariance.columns
        ),
        linkage_method=linkage_method,
        distance_metric=distance_metric,
        plot_dendrogram=plot_dendrogram,
    )

    nested_clusters = cluster_traverser(
        linkage_matrix, labels=covariance.index.tolist()
    )

    return top_down_allocation(
        nested_clusters=nested_clusters,
        covariance=covariance,
        get_cluster_var=get_cluster_var,
    )
