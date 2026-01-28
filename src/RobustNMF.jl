"""
RobustNMF

A Julia module for Non-negative Matrix Factorization (NMF) with utilities for
data generation, noise corruption, normalization, and image loading.

# Overview
This package provides tools for experimenting with **standard and robust NMF**
methods. It includes utilities for generating synthetic non-negative data,
corrupting data with Gaussian noise or sparse outliers, preprocessing data,
and loading image datasets into matrix form suitable for NMF.

# Features
- **Synthetic Data Generation**: Create low-rank non-negative matrices.
- **Noise Models**:
  - Additive Gaussian noise
  - Sparse positive outliers
- **Preprocessing Utilities**: Non-negativity enforcement and normalization.
- **Image Loading**: Load and vectorize grayscale images from folders.
- **Standard NMF**: Factorization and reconstruction utilities.

# Exports
- **Data Utilities**:
  - `generate_synthetic_data`
  - `add_gaussian_noise!`
  - `add_sparse_outliers!`
  - `normalize_nonnegative!`
  - `load_image_folder`

- **NMF Algorithms**:
  - `nmf`
  - `X_reconstruct`

# Examples
```jldoctest
julia> using RobustNMF

julia> X, W, H = generate_synthetic_data(50, 40; rank=5, seed=1);

julia> size(X)
(50, 40)

julia> add_gaussian_noise!(X; σ=0.05);

julia> normalize_nonnegative!(X);
```
"""

module RobustNMF

include("Data.jl")
include("StandardNMF.jl")
include("RobustNMFAlgorithms.jl")
include("Plotting.jl")

export
generate_synthetic_data,
add_gaussian_noise!,
add_sparse_outliers!,
normalize_nonnegative!,
load_image_folder,
nmf,
l21_loss, 
robustnmf_l21,
robustnmf_huber,
robustnmf,
plot_basis_vectors, plot_activation_coefficients, 
plot_reconstruction_comparison, plot_convergence,
plot_nmf_summary, plot_image_reconstruction

#QUESTION l21_loss not used? just in example in RobustNMFAlgorithms
end # module RobustNMF

