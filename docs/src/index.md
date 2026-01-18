# Getting Started with RobustNMF.jl

## Installation

Clone the repository and activate the project environment.
The following steps work when executed from the repository root directory.

```julia
# Enter the Julia package manager
]

# Activate the local project environment
activate .

# Install all required dependencies
instantiate
```

## Basic Usage

Import the package:

```julia
using RobustNMF
```

## Simple Example

Perform robust non-negative matrix factorization:

```julia

# Generate synthetic non-negative data
X, W, H = generate_synthetic_data(20, 30)

# Add Gaussian noise (in-place)
add_gaussian_noise!(X, σ = 0.2)

# Add sparse outliers (in-place)
add_sparse_outliers!(X, fraction = 0.04, magnitude = 5.2)

# Normalize and rescale data to non-negative range
normalize_nonnegative!(X)

# Run standard NMF
W_nmf, H_nmf, history = nmf(X, rank = 12, maxiter = 3000, tol = 1e-6)

# Reconstruct the data matrix (X)
X_rec = W_nmf * H_nmf

# Run robust NMF
W_robust, H_robust, history_robust = robust_nmf(X, rank = 12, maxiter = 3000, tol = 1e-6)

```

---

## Notes

-   All input data must be non-negative.

-   Functions with a ! modify their input in-place.

-   The reconstructed matrix X_rec approximates the original data X.
