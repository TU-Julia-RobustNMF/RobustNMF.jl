# RobustNMF

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/dev/)
[![Build Status](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/TU-Julia-RobustNMF/RobustNMF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TU-Julia-RobustNMF/RobustNMF.jl)

## Overview

**Robust Non-negative Matrix Factorization (NMF)** in Julia for data with noise and outliers.

This package provides two complementary algorithms:

- **Standard NMF** - Optimized for clean data using L2 (Frobenius) loss
- **Robust NMF (Huber)** - Robust to outliers using Huber loss with IRLS updates

The **Huber loss** combines the best of both worlds:

- Small errors: quadratic (precise, like L2)
- Large errors: linear (robust, like L1)

### Key Features

- **Two NMF algorithms** - Standard and Robust (Huber loss)
- **Data utilities** - Synthetic data generation, noise/outlier injection, normalization
- **Visualization** - Basis vectors, reconstructions, convergence tracking, and summaries
- **Easy comparison** - Evaluate both algorithms on the same data

---

## Installation

Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/TU-Julia-RobustNMF/RobustNMF.jl")
using RobustNMF
```

Alternatively, use the package manager:

```julia
]
add https://github.com/TU-Julia-RobustNMF/RobustNMF.jl.git
```

**Requirements:** Julia 1.11+ (see `Project.toml`)

---

## Quick Start

### Example 1: Generate Data with Outliers

```julia
using RobustNMF

# Generate clean synthetic data
X_clean, W_true, H_true = generate_synthetic_data(100, 60; rank=10, seed=42)

# Create corrupted version with outliers
X_outliers = copy(X_clean)
add_sparse_outliers!(X_outliers; fraction=0.05, magnitude=5.0, seed=42)
```

### Example 2: Train Both Algorithms on Noisy Data

```julia
# Train standard NMF on data with outliers
W_std, H_std, hist_std = nmf(X_outliers; rank=10, maxiter=500, tol=1e-5)

# Train robust NMF (Huber loss) on the same data
W_rob, H_rob, hist_rob = robustnmf(X_outliers; rank=10, maxiter=500, delta=1.0, seed=42)
```

### Example 3: Compare Performance on Clean Data

```julia
# Evaluate reconstruction error on original clean data
mae_standard = mean(abs.(X_clean - W_std*H_std))
mae_robust = mean(abs.(X_clean - W_rob*H_rob))

println("Standard NMF MAE: $mae_standard")
println("Robust NMF MAE:   $mae_robust")
println("Improvement:      $(round((mae_standard - mae_robust)/mae_standard*100, digits=1))%")
```

### Example 4: Visualize Results

```julia

# Full summary for Standard NMF
plot_nmf_summary(X_outliers, W_std, H_std, hist_std; title="Standard NMF")

# Full summary for Robust NMF
plot_nmf_summary(X_outliers, W_rob, H_rob, hist_rob; title="Robust NMF")

# Individual basis vectors
plot_basis_vectors(W_std; max_components=9, title="Standard NMF Basis")
plot_basis_vectors(W_rob; max_components=9, title="Robust NMF Basis")
```

**Note:** Functions ending with `!` modify inputs in-place (e.g., `add_sparse_outliers!`).

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

## API Quick Reference

**Algorithms**

- `nmf(X; rank, maxiter, tol, seed)` - Standard NMF with L2 loss
- `robustnmf(X; rank, maxiter, tol, delta, seed)` - Robust NMF with Huber loss

**Data Utilities**

- `generate_synthetic_data(m, n; rank, seed)` - Create test data
- `add_gaussian_noise!(X; σ)` - Add Gaussian noise
- `add_sparse_outliers!(X; fraction, magnitude, seed)` - Add outliers
- `normalize_nonnegative!(X)` - Shift and rescale to [0, 1]
- `load_image_folder(folder; img_size)` - Load image data

**Visualization**

- `plot_basis_vectors(W; ...)` - Show learned basis vectors
- `plot_convergence(history; ...)` - Track convergence
- `plot_reconstruction_comparison(X, X_recon; ...)` - Compare original vs. reconstructed
- `plot_nmf_summary(X, W, H, history; ...)` - Complete overview
- `plot_activation_coefficients(H; ...)` - Show coefficient matrix
- `plot_image_reconstruction(X, X_recon; ...)` - Image-specific visualization

For full API documentation, see [API Reference](https://tu-julia-robustnmf.github.io/RobustNMF.jl/stable/api/).

---

## Key Parameters

| Parameter | Default | Range        | Notes                                 |
| --------- | ------- | ------------ | ------------------------------------- |
| `rank`    | -       | 5-50         | Number of factors (usually 10-20)     |
| `maxiter` | 500     | 100-1000     | Maximum iterations                    |
| `tol`     | 1e-4    | 1e-6 to 1e-3 | Convergence tolerance                 |
| `delta`   | 1.0     | 0.5-2.0      | **Huber threshold** (Robust NMF only) |
| `seed`    | nothing | -            | Random seed for reproducibility       |

**About `delta` (Huber parameter):**

- Smaller values (0.5) → More robust to large outliers, but less precise
- Larger values (2.0) → More precise on clean data, but less outlier-resistant
- Start with `delta=1.0` and tune based on your data

---

## Demo

Run the full comparison demo. There might be a case where you should put the exact pathname of the file to make it work:

```julia
include("~/examples/demo_robustnmf.jl")
```

This generates plots comparing Standard NMF vs. Robust NMF on multiple datasets with varying outlier levels.

---

## Documentation

For detailed information:

- **[Full API Reference](https://tu-julia-robustnmf.github.io/RobustNMF.jl/stable/api/)** - All functions documented
- **[Getting Started Guide](https://tu-julia-robustnmf.github.io/RobustNMF.jl/stable/getting_started/)** - Step-by-step introduction

---

## Testing

To run tests:

```julia
using Pkg
Pkg.test("RobustNMF")
```

---

## Acknowledgments

This project was created with AI-assisted development using Claude (Anthropic) and ChatGPT (OpenAI).
While AI tools were sometimes used for code generation and documentation, all code and
documentation have been manually reviewed, tested, and validated by the authors
to ensure quality and correctness.

---

## License

See `LICENSE`.
