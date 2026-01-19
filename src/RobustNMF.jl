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

export
generate_synthetic_data,
add_gaussian_noise!,
add_sparse_outliers!,
normalize_nonnegative!,
load_image_folder,
nmf,
X_reconstruct,
robust_nmf

# The L2,1-norm is the row-wise L2 norms summed up. It is robust to outliers
function l21norm(X)
  return sum(norm.(eachrow(X)))
end

function update(X,F,G)
  s = first(size(X))
  d = [1 / norm(X[:,i]) - F*G[:,i] for i = 1:s]
  D = Diagonal(d)

  fn, fm = size(F)
  Fnew = zeros(Real, fn, fm)
  F1 = X * D * G'
  F2 = F * G * D * G'
  for j = 1:fn
    for k = 1:fm
      Fnew[j,k] = F[j,k] * (F1[j,k] / F2[j,k])
    end
  end

  gn,gm = size(G)
  Gnew = zeros(Real, gn, gm)
  G1 = F' * X * D
  G2 = F' * F * G * D
  for k = 1:gn
    for i = 1:gm
      Gnew[k,i] = G[k,i] * (G1[k,i] / G2[k,i])
    end
  end

  return Fnew, Gnew
end

# L2,1-NMF
function robustnmf(X; rank::Int = 10, maxiter::Int = 500, tol::Float64 = 1e-4)
  m, n = size(X)
        F = rand(m, rank)
        G = rand(rank, n)
        @assert minimum(X) >= 0 "X must be non-negative"

  # update until convergence is reached or at most maxiter times
  for iter = 1:maxiter
    F,G = update(X,F,G)

    # compute error
    error = l21norm(X - F * G)

    # break loop if sufficiently converged
    if (error < tol)
      return F,G
    else
      continue
    end
  end
  return F,G
end
end # module RobustNMF

