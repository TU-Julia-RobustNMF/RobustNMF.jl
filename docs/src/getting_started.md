# Getting Started with RobustNMF.jl

## Installation

Install via the Julia package manager using the Git URL (since the package is not registered):

```julia
]
add https://github.com/TU-Julia-RobustNMF/RobustNMF.jl.git
```

Or using `Pkg`:

```julia
using Pkg
Pkg.add(url="https://github.com/TU-Julia-RobustNMF/RobustNMF.jl")
using RobustNMF
```

**Requirements:** Julia 1.11+ (see `Project.toml`)

---

## Basic Usage

Import the package:

```julia
using RobustNMF
```

---

## Your First Example

Perform robust non-negative matrix factorization with noisy data:

```julia
# Generate synthetic non-negative data
X, W_true, H_true = generate_synthetic_data(50, 40; rank=6, seed=1)

# Add Gaussian noise (in-place)
add_gaussian_noise!(X; σ=0.2)

# Add sparse outliers (in-place)
add_sparse_outliers!(X; fraction=0.05, magnitude=5.0, seed=1)

# Normalize and rescale data to non-negative range
normalize_nonnegative!(X)

# Run standard NMF
W_nmf, H_nmf, history = nmf(X; rank=6, maxiter=500, tol=1e-4)

# Reconstruct the data matrix (X)
X_rec = W_nmf * H_nmf

# Run robust NMF
W_robust, H_robust, history_robust = robustnmf(X; rank=6, maxiter=500, tol=1e-3, seed=1)

# Compare relative reconstruction error
relerr_nmf = norm(X - W_nmf * H_nmf) / norm(X)
relerr_robust = norm(X - W_robust * H_robust) / norm(X)
println("relative error NMF:    ", relerr_nmf)
println("relative error robust: ", relerr_robust)

# Plot convergence of both runs
p_nmf = plot_convergence(history; objective=:frobenius, title="NMF Convergence")
p_rob = plot_convergence(history_robust; objective=:huber, title="Robust NMF Convergence")
display(p_nmf)
display(p_rob)
```

---

## Important Notes

- All input data must be non-negative.
- Functions with a `!` modify their input in-place (e.g., `add_sparse_outliers!`).
- The reconstructed matrix `X_rec` approximates the original data `X`.

---

## When to Use What?

| Scenario                | Algorithm                | Notes                                 |
| ----------------------- | ------------------------ | ------------------------------------- |
| Clean data              | `nmf()`                  | Fast, standard choice                 |
| Gaussian noise          | `nmf()` or `robustnmf()` | Robust NMF handles it better          |
| Sparse outliers (5-10%) | `robustnmf()`            | Recommended                           |
| Heavy outliers (>10%)   | `robustnmf()`            | Set `delta` lower for more robustness |
| Performance critical    | `nmf()`                  | Standard NMF is faster                |

---

## Next Steps

- **[Explore Examples](examples.md)** - Practical workflows and use cases
- **[Check API Reference](api.md)** - Detailed function documentation