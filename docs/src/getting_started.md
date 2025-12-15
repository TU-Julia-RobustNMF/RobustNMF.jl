# Getting Started with RobustNMF.jl

## Installation

Add the package to your Julia environment:

```julia
using Pkg
Pkg.add("RobustNMF")
```

## Basic Usage

Import the package:

```julia
using RobustNMF
```

## Simple Example

Perform robust non-negative matrix factorization on your data:

```julia
# generate data
X, W, H = generate_synthetic_data(20, 30)

# add noise
W, H = robust_nmf(X, rank=10)

# Reconstruct
X_approx = W * H
```
