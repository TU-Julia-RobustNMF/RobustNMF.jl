```@meta
CurrentModule = RobustNMF
```

# RobustNMF

Documentation for [RobustNMF](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl).

## Algorithms:
### 1. Standard NMF:
Standard non-matrix factorization helps to clean data without outliers and minimizes the Frobenius norm:
```julia
function nmf(X; rank::Int = 10, maxiter::Int = 500, tol::Float64 = 1e-4)
        m, n = size(X)
        W = rand(m, rank)
        H = rand(rank, n)
        err = norm(X - W * H)
```
**W** - Basic Matrix (m x rank) <br>
**H** - Coefficient Matrix (rank x n)<br>

### 2. RobustNMF - L2,1-norm:
RobustNMF with L2,1-norm approach is best for sample-wised outliers and data with corrupted samples (entire columns)
```julia
function robustnmf(X; rank::Int = 10, maxiter::Int = 500, tol::Float64 = 1e-4)
m, n = size(X)
        F = rand(m, rank)
        G = rand(rank, n)
        error = l21norm(X - F * G)
```
**F** - Basic Matrix (m x rank) <br>
**G** - Coefficient Matrix (rank x n)<br>
```@autodocs
Modules = [RobustNMF]
```
