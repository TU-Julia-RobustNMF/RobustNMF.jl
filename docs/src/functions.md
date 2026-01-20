```@meta
CurrentModule = RobustNMF
```

# RobustNMF

Documentation for [RobustNMF](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl).

## Algorithms

### 1. Standard NMF
Standard non-negative matrix factorization helps with clean data without outliers and minimizes the Frobenius norm.

```@docs
nmf
```

**Returns:**
- **W** - Basis Matrix (m × rank)
- **H** - Coefficient Matrix (rank × n)

### 2. Robust NMF - L2,1-norm
RobustNMF with L2,1-norm approach is best for sample-wise outliers and data with corrupted samples (entire columns).

```@docs
l21_nmf
l21_update
```

**Returns:**
- **F** - Basis Matrix (m × rank)
- **G** - Coefficient Matrix (rank × n)

**Helper functions:**
```@docs
l21norm
```

## Data Generation and Preprocessing

```@docs
generate_synthetic_data
add_gaussian_noise!
add_sparse_outliers!
normalize_nonnegative!
load_image_folder
```

## Visualization Functions

### 1. plot_convergence
The convergence plot of NMF algorithms shows if the algorithm converged, compares convergence speed between methods, and identifies if more iterations are needed.

```@docs
plot_convergence
```

### 2. plot_basis_vectors
This function helps to visualize the learned basis vectors (W).
- Each subplot shows one basis vector
- For images: Should show meaningful parts (e.g., facial features, object components)
- For text: Represents topics or themes

```@docs
plot_basis_vectors
```

**Example:**
```julia
W, H, _ = nmf(X; rank=10)
plot_basis_vectors(W; max_components=10)

# For image data with known dimensions
plot_basis_vectors(W; img_shape=(28, 28), max_components=16)
```

### 3. plot_reconstruction_comparison
Shows original data vs. reconstructed data side-by-side. We can visually evaluate reconstruction accuracy and see where algorithm struggles.

```@docs
plot_reconstruction_comparison
```

**Example:**
```julia
W, H, _ = nmf(X; rank=10)
X_recon = W * H
plot_reconstruction_comparison(X, X_recon; img_shape=(28, 28), n_samples=6)
```

### 4. plot_nmf_summary
Creates a comprehensive summary with basis vectors, reconstructions, and convergence.

```@docs
plot_nmf_summary
```

**Example:**
```julia
W, H, history = nmf(X; rank=10)
plot_nmf_summary(X, W, H, history; img_shape=(28, 28))
```

### 5. Other Visualization Functions

```@docs
plot_activation_coefficients
plot_image_reconstruction
```

## Performance Metrics

### RMSE (Root Mean Square Error)
- Measures average reconstruction error
- Lower is better
- Standard NMF optimizes this metric

```julia
rmse = sqrt(mean((X - W*H).^2))
```

### MAE (Mean Absolute Error)
- Measures average absolute reconstruction error
- Lower is better
- Better metric for comparing robustness

```julia
mae = mean(abs.(X - W*H))
```

### Relative Error
- Error as percentage of data magnitude
- Lower is better
- Typical range values: 1-20% approximately

```julia
rel_error = norm(X - W*H) / norm(X)
```

## Running the Demo

To run the demo version of RobustNMF:

```julia
# Instantiate your package
using Pkg
Pkg.instantiate()

# Run the following command to have the demo version of RobustNMF
include("examples/demo_robustnmf.jl")
```

Afterwards, you will see the plots that compare Standard NMF and Robust NMF, along with performance metrics such as RMSE and MAE.
