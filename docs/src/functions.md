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
```
**W** - Basic Matrix (m x rank) <br>
**H** - Coefficient Matrix (rank x n)<br>
**history** - Vector of Frobenius errors at each iteration

### 2. RobustNMF - L2,1-norm:
RobustNMF with L2,1-norm approach is best for sample-wised outliers and data with corrupted samples (entire columns)


```@autodocs
Modules = [RobustNMF]
```
