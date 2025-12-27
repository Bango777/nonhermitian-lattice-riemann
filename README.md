# Spectral Emergence of Riemann Statistics in a Non-Hermitian Lattice

This repository contains the numerical code and data associated with the study of spectral statistics in a three-dimensional non-Hermitian lattice model inspired by Heisenberg-like commutation structures.

The goal of the project is to investigate whether the normalized level-spacing statistics of the imaginary parts of the complex eigenvalues of a non-Hermitian lattice operator reproduce the statistical properties of the non-trivial zeros of the Riemann zeta function.

---

## Contents

- `fase24.py`  
  Python script implementing **Phase 24**, consisting of a parameter scan over lattice size `N` and non-Hermiticity parameter `epsilon`.  
  For each parameter pair, the script computes the Kolmogorov–Smirnov distance between:
  - normalized level spacings of the lattice spectrum
  - normalized spacings of the imaginary parts of Riemann zeros

- `zeri_riemann_primi_1000.txt`  
  Text file containing the imaginary parts of the first non-trivial zeros of the Riemann zeta function, used as reference data.

---

## Requirements

- Python ≥ 3.8  
- NumPy  
- Matplotlib  
- SciPy (optional, but recommended for the Kolmogorov–Smirnov test)

---

## Usage

Clone the repository and run:

```bash
python fase24.py
The file `zeri_riemann_primi_1000.txt` must be located in the same directory as the script.
