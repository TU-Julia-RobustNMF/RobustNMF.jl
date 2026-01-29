# RobustNMF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/dev/)
[![Build Status](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/TU-Julia-RobustNMF/RobustNMF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TU-Julia-RobustNMF/RobustNMF.jl)

This project implements **Non-negative Matrix Factorization (NMF)** methods in Julia, with a focus on robustness against noise and outliers.

The package provides:

-   a standard NMF implementation using an L2 (Frobenius) loss
-   a robust NMF implementation using an L2,1 loss for outlier resistance
-   utilities for synthetic data generation and preprocessing
-   tools to add noise and sparse outliers for robustness evaluation
-   visualization helpers for basis vectors, reconstructions, and convergence

---

## Features

-   **Standard NMF**

    -   Multiplicative update rules
    -   Frobenius (L2) reconstruction loss
    -   Convergence monitoring via error history

-   **Robust NMF (L2,1)**

    -   Robust to sample-wise (column) outliers
    -   Multiplicative updates with L2,1 reconstruction loss
    -   Convergence tracking via L2,1 error history

-   **Data preparation utilities**
    -   Synthetic non-negative data generation
    -   Gaussian noise injection
    -   Sparse outlier corruption
    -   Normalization to non-negative ranges
    -   Loading and preprocessing of image datasets

-   **Visualization**
    -   Basis vectors and activation heatmaps
    -   Reconstruction comparisons
    -   Convergence curves and summary dashboards

---

## Installation

Install via the Julia package manager using the Git URL (since the package is not registered):

```julia
]
add https://github.com/TU-Julia-RobustNMF/RobustNMF.jl.git
```

Julia version: `1.11` (see `Project.toml`).

---

## Quick Start

```julia
using RobustNMF

# Generate synthetic non-negative data
X, W_true, H_true = generate_synthetic_data(50, 40; rank=6, seed=1)

# Corrupt data (in-place)
add_gaussian_noise!(X; sigma=0.1)
add_sparse_outliers!(X; fraction=0.05, magnitude=5.0, seed=1)
normalize_nonnegative!(X)

# Standard NMF
W_nmf, H_nmf, hist_nmf = nmf(X; rank=6, maxiter=500, tol=1e-4)

# Robust NMF (L2,1)
W_rob, H_rob, hist_rob = robustnmf(X; rank=6, maxiter=500, tol=1e-3, seed=1)
```

Note: Functions ending with `!` modify inputs in-place.

---

## API at a Glance

-   **Algorithms**
    -   `nmf` (standard L2 / Frobenius)
    -   `robustnmf` (L2,1 robust)

-   **Data utilities**
    -   `generate_synthetic_data`
    -   `add_gaussian_noise!`
    -   `add_sparse_outliers!`
    -   `normalize_nonnegative!`
    -   `load_image_folder`

-   **Visualization**
    -   `plot_basis_vectors`
    -   `plot_activation_coefficients`
    -   `plot_reconstruction_comparison`
    -   `plot_convergence`
    -   `plot_nmf_summary`
    -   `plot_image_reconstruction`

---

## Demo

Run the comparison demo:

```julia
include("examples/demo_robustnmf.jl")
```

It generates multiple plots that compare standard and robust NMF on clean and corrupted data.

---

## Tests

```julia
using Pkg
Pkg.test()
```

---

## Documentation

Build the docs locally from the repository root:

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

---

## License

See `LICENSE`.
