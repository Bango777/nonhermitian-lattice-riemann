#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PHASE 24 – PARAMETER MAP (N, epsilon)
Heisenberg-like non-Hermitian 3D lattice vs Riemann zeros

Purpose:
  - Explore different values of N (lattice size) and epsilon
    (non-Hermiticity) and measure how well the normalized gaps of the
    lattice reproduce the normalized gaps of the Riemann zeros.
  - For each pair (N, epsilon) we compute:
        D_KS = Kolmogorov–Smirnov distance between the two distributions.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("WARNING: SciPy not found. Using a simplified KS test.\n")


# ------------------------------------------------------------
#  Loading Riemann zeros
# ------------------------------------------------------------

def carica_zeri_riemann(filename="zeri_riemann_primi_1000.txt",
                        n_max=400):
    """
    Load from file the gamma_n values (imaginary parts of the Riemann zeros
    on the critical line 1/2 + i*gamma_n).

    The file must contain one gamma_n per line.
    """
    try:
        data = np.loadtxt(filename, dtype=float)
    except OSError:
        raise SystemExit(
            f"ERROR: unable to read file '{filename}'. "
            "Make sure it is in the same folder as this script."
        )

    if data.ndim == 0:
        data = np.array([data], dtype=float)

    gamma = np.sort(data[:n_max])
    return gamma


# ------------------------------------------------------------
#  3D lattice construction and non-Hermitian dynamic matrix
# ------------------------------------------------------------

def indice_lin(i, j, k, N_int):
    """Linear index for a 3D lattice N_int x N_int x N_int."""
    return i + N_int * (j + N_int * k)


def costruisci_matrici_base_reticolo(N):
    """
    Build:
      - L : real symmetric 3D Laplacian matrix (elastic couplings)
      - K : real antisymmetric part (Heisenberg-commutator-like)
    for a 3D lattice with N_int = N - 2 internal nodes per side.

    Returns (L, K, N_int)
    """
    if N <= 2:
        raise ValueError("N must be > 2 to have internal nodes.")

    N_int = N - 2
    M = N_int ** 3

    L = np.zeros((M, M), dtype=float)
    K = np.zeros((M, M), dtype=float)

    # Build the Laplacian L (nearest-neighbor couplings only)
    for i in range(N_int):
        for j in range(N_int):
            for k in range(N_int):
                p = indice_lin(i, j, k, N_int)
                deg = 0

                # All Cartesian neighbors (+/- x, +/- y, +/- z)
                for dx, dy, dz in [(1, 0, 0), (-1, 0, 0),
                                   (0, 1, 0), (0, -1, 0),
                                   (0, 0, 1), (0, 0, -1)]:
                    ix, jy, kz = i + dx, j + dy, k + dz
                    if (0 <= ix < N_int and
                            0 <= jy < N_int and
                            0 <= kz < N_int):
                        q = indice_lin(ix, jy, kz, N_int)
                        L[p, q] = 1.0
                        deg += 1

                L[p, p] = -float(deg)

    # Build the antisymmetric part K:
    # to avoid double counting, consider only positive directions.
    for i in range(N_int):
        for j in range(N_int):
            for k in range(N_int):
                p = indice_lin(i, j, k, N_int)

                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    ix, jy, kz = i + dx, j + dy, k + dz
                    if (0 <= ix < N_int and
                            0 <= jy < N_int and
                            0 <= kz < N_int):
                        q = indice_lin(ix, jy, kz, N_int)
                        # Antisymmetric coupling: K = -K^T
                        K[p, q] += 1.0
                        K[q, p] -= 1.0

    return L, K, N_int


def matrice_non_hermitiana(L, K, epsilon):
    """
    Combine the two real matrices into a non-Hermitian complex matrix:
        M = L + i * epsilon * K
    """
    return L.astype(complex) + 1j * epsilon * K


# ------------------------------------------------------------
#  Spectral analysis and gaps
# ------------------------------------------------------------

def gap_normalizzati(seq):
    """
    Given a one-dimensional sorted array (seq),
    compute the normalized gaps s_n = (x_{n+1} - x_n) / <gap>.
    """
    seq = np.asarray(seq, dtype=float)
    seq = np.sort(seq)
    gaps = np.diff(seq)
    # remove negative or null gaps for safety
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return np.array([], dtype=float)

    s = gaps / np.mean(gaps)
    return s


def ks_distance(x, y):
    """
    KS distance between two 1D sets.
    If SciPy is available we use ks_2samp, otherwise
    we implement a simple version.
    """
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))

    if len(x) == 0 or len(y) == 0:
        return np.nan

    if HAVE_SCIPY:
        return ks_2samp(x, y).statistic

    # Minimal version without SciPy
    data_all = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, data_all, side="right") / len(x)
    cdf_y = np.searchsorted(y, data_all, side="right") / len(y)
    D = np.max(np.abs(cdf_x - cdf_y))
    return float(D)


# ------------------------------------------------------------
#  Phase 24 – parameter scan (N, epsilon)
# ------------------------------------------------------------

def fase24():
    print("=== PHASE 24: PARAMETER MAP (N, epsilon) – lattice vs Riemann zeros ===")

    # Scan parameters (you can modify them for deeper exploration)
    list_N = [9, 11, 13]                 # total lattice side (N_int = N-2)
    list_eps = [0.05, 0.10, 0.20, 0.30]  # degree of non-Hermiticity

    # Number of Riemann zeros to use
    n_zeri = 400

    # Load Riemann zeros
    gamma_riem = carica_zeri_riemann(n_max=n_zeri)

    # Precompute normalized gaps of the zeros
    s_riem = gap_normalizzati(gamma_riem)

    # Matrix to store KS distances
    Dmat = np.zeros((len(list_N), len(list_eps)), dtype=float)

    print(f"Riemann zeros used: {len(gamma_riem)}")
    print("")

    # Parameter loop
    for iN, N in enumerate(list_N):
        print(f"Building base matrices for N = {N} ...")
        L, K, N_int = costruisci_matrici_base_reticolo(N)
        M_dim = N_int ** 3
        print(f"  -> N_int = {N_int}, matrix size = {M_dim} x {M_dim}")

        for jE, eps in enumerate(list_eps):
            print(f"  Analysis for epsilon = {eps:.3f} ...", end="", flush=True)

            M = matrice_non_hermitiana(L, K, eps)

            # Complex eigenvalues
            vals = np.linalg.eigvals(M)

            # Use only those with Im(lambda) > 0
            gamma_ret = np.sort(np.imag(vals))
            gamma_ret = gamma_ret[gamma_ret > 0]

            # Align the number of levels used
            n_levels = min(len(gamma_ret), len(gamma_riem))
            if n_levels < 10:
                print(" few levels, skipping.")
                D = np.nan
            else:
                gamma_ret = gamma_ret[:n_levels]
                gamma_r = gamma_riem[:n_levels]

                s_ret = gap_normalizzati(gamma_ret)
                s_r = gap_normalizzati(gamma_r)

                n_gaps = min(len(s_ret), len(s_r))
                if n_gaps < 10:
                    print(" few gaps, skipping.")
                    D = np.nan
                else:
                    s_ret = s_ret[:n_gaps]
                    s_r = s_r[:n_gaps]
                    D = ks_distance(s_ret, s_r)
                    print(f" D_KS = {D:.4f}")

            Dmat[iN, jE] = D

        print("")

    # --------------------------------------------------------
    # Print summary table
    # --------------------------------------------------------
    print("=== OVERALL RESULTS – PHASE 24 ===")
    header = "N  epsilon   D_KS"
    print(header)
    print("-" * len(header))
    for iN, N in enumerate(list_N):
        for jE, eps in enumerate(list_eps):
            D = Dmat[iN, jE]
            if np.isnan(D):
                txtD = "nan"
            else:
                txtD = f"{D:.4f}"
            print(f"{N:2d}  {eps:7.3f}  {txtD}")
    print("")

    # Find the pair with minimum D_KS
    if np.all(np.isnan(Dmat)):
        print("WARNING: all KS distances are NaN (data issue).")
        return

    mask = ~np.isnan(Dmat)
    i_best, j_best = np.where(Dmat == np.nanmin(Dmat[mask]))
    i_best, j_best = int(i_best[0]), int(j_best[0])
    N_best = list_N[i_best]
    eps_best = list_eps[j_best]
    D_best = Dmat[iN, jE] if False else Dmat[i_best, j_best]  # keep explicit
    print(f"Best agreement: N = {N_best}, epsilon = {eps_best:.3f}, D_KS = {D_best:.4f}")

    # --------------------------------------------------------
    # Plot D_KS(N, epsilon) map
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        Dmat,
        origin="lower",
        aspect="auto",
        extent=[min(list_eps) - 0.5 * (list_eps[1] - list_eps[0]),
                max(list_eps) + 0.5 * (list_eps[1] - list_eps[0]),
                min(list_N) - 0.5 * (list_N[1] - list_N[0]),
                max(list_N) + 0.5 * (list_N[1] - list_N[0])],
        cmap="viridis"
    )
    ax.set_xlabel("epsilon (non-Hermiticity)")
    ax.set_ylabel("N (lattice side)")
    ax.set_title("Phase 24 – KS distance between lattice gaps and Riemann zeros")

    # Highlight the minimum
    ax.scatter([eps_best], [N_best], color="red", marker="o", s=80,
               label=f"min D_KS = {D_best:.3f}")
    ax.legend(loc="upper right")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("KS distance")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------

if __name__ == "__main__":
    fase24()
