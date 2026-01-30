# RobustNMF.jl

Welcome to **RobustNMF.jl** - Robust Non-negative Matrix Factorization in Julia for data with noise and outliers.

## Overview

This package provides two complementary algorithms for non-negative matrix factorization:

- **Standard NMF** - Optimized for clean data using L2 (Frobenius) loss
- **Robust NMF (Huber)** - Robust to outliers using Huber loss with IRLS updates

The **Huber loss** combines the best of both worlds:
- Small errors: quadratic (precise, like L2)
- Large errors: linear (robust, like L1)

## Key Features

- **Two NMF algorithms** - Standard and Robust (Huber loss) options
- **Data utilities** - Synthetic data generation, noise/outlier injection, normalization, image loading
- **Visualization** - Basis vectors, reconstructions, convergence tracking, and comprehensive summaries
- **Easy comparison** - Evaluate both algorithms on the same data

---

## Quick Navigation

- **[Getting Started](getting_started.md)** - Installation and first example
- **[API Reference](api.md)** - Complete function documentation
- **[Examples](examples.md)** - Practical use cases and workflows

---

## Installation

Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/TU-Julia-RobustNMF/RobustNMF.jl")
using RobustNMF
```

**Requirements:** Julia 1.11+ (see `Project.toml`)

---

For more information, see the [GitHub repository](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl).
