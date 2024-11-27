import numpy as np
from scipy.linalg import svd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import sparse_encode
import seaborn as sns
import logging
import matplotlib.pyplot as plt

Logger = logging.getLogger(__name__)


def generate_matrix(n_prot, n_total_pept, p_dense):
    """
    Generate a matrix of shape (n_prot, n_total_pept) where p_dense percent of the elements
    are random values between 0 and 1, and the rest are 0.

    Parameters:
    n_prot (int): Number of rows (proteins) in the matrix
    n_total_pept (int): Number of columns (total peptides) in the matrix
    p_dense (float): Percentage of elements that should be non-zero (between 0 and 1)

    Returns:
    numpy.ndarray: The generated matrix
    """
    # Calculate the number of non-zero elements
    n_non_zero = int(n_prot * n_total_pept * p_dense)

    # Create a matrix of zeros
    matrix = np.zeros((n_total_pept, n_prot))

    # Select random indices for the non-zero elements
    non_zero_indices = np.random.choice(
        n_prot * n_total_pept, size=n_non_zero, replace=False
    )

    # Assign random values between 0 and 1 to the non-zero elements
    matrix.flat[non_zero_indices] = np.random.rand(n_non_zero)

    return matrix


def generate_vector(n_prot, raw=True):
    """
    Generate a vector of shape (n_prot) with random values ranging from 0 to 10^9.

    Parameters:
    n_prot (int): The number of elements in the vector

    Returns:
    numpy.ndarray: The generated vector
    """
    vec = np.random.normal(loc=6, scale=0.75, size=n_prot)
    if raw:
        vec = 10**vec
    return vec


def simulate_noisy_protein_fingerprint(
    n_prot, n_total_pept, p_dense, signal_to_noise_ratio
):
    """
    Simulate a noisy protein_fingerprint matrix.

    Parameters:
    n_total_pept (int): Number of peptides
    n_prot (int): Number of proteins
    signal_to_noise_ratio (float): Ratio of signal to noise, higher values mean less noise

    Returns:
    numpy.ndarray: The simulated noisy protein_fingerprint matrix, shape (n_total_pept, n_prot)
    """
    # Generate a base protein_fingerprint matrix
    base_protein_fingerprint = generate_matrix(n_prot, n_total_pept, p_dense)

    # Calculate the signal power
    signal_power = np.var(base_protein_fingerprint)

    # Calculate the noise power based on the signal-to-noise ratio
    noise_power = signal_power / signal_to_noise_ratio

    # Generate the noise matrix
    noise = np.random.normal(0, np.sqrt(noise_power), size=(n_total_pept, n_prot))

    # Add the noise to the base protein_fingerprint matrix
    noisy_protein_fingerprint = base_protein_fingerprint + noise

    return base_protein_fingerprint, noisy_protein_fingerprint


def infer_protein_quant(pept_act, protein_fingerprint, algorithm="lasso_lars"):
    """
    Infer the protein_fingerprint matrix given pept_act and protein_quant using sparse_encode.

    Parameters:
    pept_act (numpy.ndarray): Vector of peptide activities, shape (n_total_pept,)
    protein_quant (numpy.ndarray): Vector of protein quantities, shape (n_prot,)

    Returns:
    numpy.ndarray: The inferred protein_fingerprint matrix, shape (n_total_pept, n_prot)
    """

    # Normalize protein_quant vector
    # protein_fingerprint = protein_fingerprint / np.linalg.norm(protein_fingerprint)

    # Use sparse_encode to infer protein_fingerprint
    protein_quant = sparse_encode(
        pept_act.reshape(-1, 1).T,
        protein_fingerprint.T,
        positive=True,
        algorithm=algorithm,
        alpha=0,
    )

    return protein_quant.T


def k_svd(
    pept_act, initial_protein_fingerprint, sparsity_param, max_iterations=100, tol=1e-6
):
    """
    Implement the K-SVD algorithm to jointly optimize the protein_fingerprint matrix and protein_quant vector given pept_act.

    Parameters:
    pept_act (numpy.ndarray): The peptide activity vector, shape (n_total_pept,)
    initial_protein_fingerprint (numpy.ndarray): The initial protein_fingerprint matrix, shape (n_total_pept, n_prot)
    sparsity_param (int): The desired sparsity level for the protein_quant vector
    max_iterations (int): Maximum number of K-SVD iterations
    tol (float): Tolerance for convergence

    Returns:
    numpy.ndarray: The optimized protein_fingerprint matrix, shape (n_total_pept, n_prot)
    numpy.ndarray: The optimized protein_quant vector, shape (n_prot,)
    """
    n_total_pept, n_prot = initial_protein_fingerprint.shape

    protein_fingerprint = initial_protein_fingerprint.copy()
    protein_quant = np.zeros(n_prot)

    prev_error = np.inf

    for iteration in range(max_iterations):
        # Sparse Coding Step
        for i in range(n_total_pept):
            protein_quant_i, _ = orthogonal_matching_pursuit(
                protein_fingerprint[i, :], pept_act[i], sparsity_param
            )
            protein_quant[i] = protein_quant_i

        # Dictionary Update Step
        for j in range(n_prot):
            # Find the indices of data points that use the j-th dictionary atom
            active_indices = np.nonzero(protein_quant[:, j])[0]

            if len(active_indices) > 0:
                # Update the j-th dictionary atom
                X = pept_act[active_indices]
                D = protein_fingerprint[active_indices]
                _, _, Vt = svd(
                    X - np.dot(D, np.diag(protein_quant[active_indices, j])),
                    full_matrices=False,
                )
                protein_fingerprint[:, j] = Vt[0]

        # Calculate the reconstruction error
        pept_act_reconstructed = np.dot(protein_fingerprint, protein_quant)
        error = np.mean((pept_act - pept_act_reconstructed) ** 2)

        # Check for convergence
        if abs(prev_error - error) < tol:
            break

        prev_error = error

    return protein_fingerprint, protein_quant


def orthogonal_matching_pursuit(dictionary, signal, sparsity_param):
    """
    Implement the Orthogonal Matching Pursuit (OMP) algorithm for sparse coding.

    Parameters:
    dictionary (numpy.ndarray): The dictionary matrix, shape (n, d)
    signal (numpy.ndarray): The input signal, shape (n,)
    sparsity_param (int): The desired sparsity level for the sparse code

    Returns:
    numpy.ndarray: The sparse code, shape (d,)
    """
    d = dictionary.shape
    residual = signal
    support = []
    coefficients = np.zeros(d)

    for _ in range(sparsity_param):
        # Find the dictionary atom that is most correlated with the residual
        atom_idx = np.argmax(np.abs(np.dot(dictionary.T, residual)))

        # Add the atom index to the support
        support.append(atom_idx)

        # Solve the least-squares problem to find the coefficients
        coefs, _, _, _ = np.linalg.lstsq(dictionary[support], signal, rcond=None)
        coefficients[support] = coefs

        # Update the residual
        residual = signal - np.dot(dictionary[:, support], coefs)

    return coefficients, support


def eval_protein_quant(inferred_protein_quant, true_protein_quant):
    """
    Evaluate the inferred protein_quant vector against the true protein_quant vector.

    Parameters:
    protein_quant (numpy.ndarray): The inferred protein_quant vector, shape (n_prot,)
    true_protein_quant (numpy.ndarray): The true protein_quant vector, shape (n_prot,)

    Returns:
    float: The mean squared error between the inferred and true protein_quant vectors
    """
    # Convert inputs to shape (x,)
    if len(inferred_protein_quant.shape) > 1:
        inferred_protein_quant = inferred_protein_quant.flatten()
    if len(true_protein_quant.shape) > 1:
        true_protein_quant = true_protein_quant.flatten()
    # Calculate Spearman correlation
    corr, _ = spearmanr(true_protein_quant, inferred_protein_quant)
    Logger.info("Spearman correlation: %s", corr)
    pcorr, _ = pearsonr(true_protein_quant, inferred_protein_quant)
    Logger.info("Pearson correlation: %s", pcorr)
    sns.scatterplot(
        x=np.log10(true_protein_quant + 1), y=np.log10(1 + inferred_protein_quant)
    )
    # sns.scatterplot(x=protein_quant, y=inferred_protein_quant[:, 0])
    plt.xlabel("True Quant (log10)")
    plt.ylabel("Infer Qunat (log10)")
