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
# Your data matrix
X = rand(100, 50)

# Initialize and run factorization
W, H = robust_nmf(X, rank=10)

# Reconstruct
X_approx = W * H
```
