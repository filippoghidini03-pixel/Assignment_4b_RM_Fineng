import numpy as np
import pandas as pd
from utilities.covariance_utilities import covariance_to_correlation


def principal_component_analysis(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a matrix, returns the eigenvalues vector and the eigenvectors matrix.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sorting from greatest to lowest the eigenvalues and the eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]

    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


def detone(corr_matrix: np.ndarray, components_num: int = 1) -> np.ndarray:
    """
    Remove components_num principal components from an input correlation matrix.

    The market component of a correlation matrix is typically captured by its
    first eigenvector, which loads roughly equally on every asset. This pervasive
    common factor pulls all pairwise correlations toward positive values, washing
    out the genuine sector/idiosyncratic structure that clustering tries to recover.

    Detoning subtracts the rank-`components_num` reconstruction of C (the outer
    products of the dominant eigenvectors, scaled by their eigenvalues) and then
    rescales the diagonal back to 1 so the result is a proper correlation matrix.

    Parameters:
        corr_matrix (np.ndarray): Correlation matrix to be detoned. Shape (N, N).
        components_num (int): Number of leading principal components to remove.
            Default is 1 (remove the market factor only).

    Returns:
        np.ndarray: Detoned correlation matrix with unit diagonal. Shape (N, N).
    """


    # Step 1 — decompose the correlation matrix; eigenvalues already sorted
    #          from largest to smallest by principal_component_analysis.
    eigenvalues, eigenvectors = principal_component_analysis(corr_matrix)

    # Step 2 — select the dominant `components_num` eigenvectors and eigenvalues.
    U_m = eigenvectors[:, :components_num]           
    lam_m = eigenvalues[:components_num]            

    # Step 3 — reconstruct the market component and subtract it.
    market_component = (U_m * lam_m) @ U_m.T        
    detoned_corr = corr_matrix - market_component    

    # Step 4 — rescale so that every diagonal element equals 1.
    diag_sqrt = np.sqrt(np.diag(detoned_corr))      
    detoned_corr = detoned_corr / np.outer(diag_sqrt, diag_sqrt)

    return detoned_corr


def align_eigenvectors_to_previous(
    current_eig_vecs: pd.DataFrame,
    previous_eig_vecs: pd.DataFrame | None,
) -> pd.DataFrame:
    aligned = current_eig_vecs.copy()

    if previous_eig_vecs is None:
        if aligned.iloc[:, 0].sum() < 0:
            aligned.iloc[:, 0] *= -1
        return aligned

    common_assets = aligned.index.intersection(previous_eig_vecs.index)
    components_num = min(
        len(common_assets),
        aligned.shape[1],
        previous_eig_vecs.shape[1],
    )
    for pc in range(components_num):
        similarity = float(
            aligned.loc[common_assets, pc].dot(previous_eig_vecs.loc[common_assets, pc])
        )
        if similarity < 0:
            aligned.iloc[:, pc] *= -1

    return aligned


def pca_denoise_covariance(
    cov_matrix: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Denoise a covariance matrix by keeping k signal eigenvalues and averaging the rest.

    Args:
        cov_matrix: Sample covariance matrix (N x N).
        k: Number of principal components to retain as signal.

    Returns:
        Denoised covariance matrix and its eigenvalues (sorted descending).
    """
    eig_vals, eig_vecs = principal_component_analysis(cov_matrix)
    noise_avg = np.mean(eig_vals[k:])
    denoised_eig_vals = np.concatenate(
        [eig_vals[:k], np.full(len(eig_vals) - k, noise_avg)]
    )
    denoised_cov = (eig_vecs * denoised_eig_vals) @ eig_vecs.T
    return denoised_cov, denoised_eig_vals
