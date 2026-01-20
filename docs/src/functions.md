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

## Visualization functions:

### 1.plot_convergence:
The convergence plot of NMF algorithms shows if algorithm covereged, compares convergence speed between methods <br>
and identifies if more iterations are needed.
```julia
plot_convergence(history::Vector; title::String="NMF Convergence",
                    ylabel::String="Frobenius Error", log_scale::Bool=true)
```
### 2.plot_basis_vectors:
This function helps to visualize the learned basic vectors (W). <br>
Each subplot shows one basis vector. <br>
For images: Should show meaningful parts (e.g., facial features, object components). <br>
For text: Represents topics or themes. <br>
```julia
plot_convergence(history::Vector; title::String="NMF Convergence",
                    ylabel::String="Frobenius Error", log_scale::Bool=true)
```julia
W, H, _ = nmf(X; rank=10)
plot_basis_vectors(W; max_components=10)

# For image data with known dimensions
plot_basis_vectors(W; img_shape=(28, 28), max_components=16)
```
### 3.plot_reconstruction_comparison:
Shows original data vs. reconstructed data side-by-side. <br>
We can visually evaluate reconstruction accuracy and see where algorithm struggles.

```julia
W, H, _ = nmf(X; rank=10)
X_recon = W * H
plot_reconstruction_comparison(X, X_recon; img_shape=(28, 28), n_samples=6)
```

```@autodocs
Modules = [RobustNMF]
```
### 4.plot_nmf_summary:
Creates a comprehensive summary with basis vectors, reconstructions, and convergence.
```julia
W, H, history = nmf(X; rank=10)
plot_nmf_summary(X, W, H, history; img_shape=(28, 28))
```
## Performance metrics:
### RMSE (Root Mean Square Error):
 - Measures average reconstruction error
 - Lower is better
 - Standard NMF optimizes this metric
```julia
rmse = sqrt(mean((X - W*H).^2))
```
### MAE (Mean Absolute Error):
 - Measures average absolute reconstruction error
 - Lower is better
 - Better metric for comparing robustness
```julia
mae = mean(abs.(X - W*H))
```

### Relative Error:
 - Error as percentage of data magnitude 
 - Lower is better
 - Typically range values: 1-20% approximately
```julia
rel_error = norm(X - W*H) / norm(X)
```

## Demo RobustNMF:
To run the demo version of RobustNMF, it is better to run the following:
```julia
#Instantiate your package
>julia Pkg.instantiate()

#Run the following command to have the demo version of RobustNMF
>julia include("examples/demo_robustnmf.jl")
```
Afterwards, you will see the plots that compare the StandardNMF and RobustNMF, <br>
having also performance metrics such as RMSE and MAE.






