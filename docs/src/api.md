```@meta
CurrentModule = RobustNMF
```

# API Reference

Complete documentation of all RobustNMF.jl functions.

---

## Algorithms

### Standard NMF

Standard non-negative matrix factorization optimized for clean data without outliers. Uses the Frobenius norm (L2 loss).

```@docs
nmf
```

---

### Robust NMF (Huber Loss)

Robust NMF using the Huber loss and IRLS-weighted multiplicative updates. More resistant to outliers than standard NMF.

```@docs
robustnmf
robustnmf_huber
```

---

### Robust NMF (Legacy L2,1)

Legacy L2,1-robust NMF (kept for compatibility).

```@docs
robustnmf_l21
```

---

### Helper Functions (Internal)

```@docs
update_huber
huber_loss
huber_weights
update_l21
l21_loss
```

---

## Data Generation and Preprocessing

Utilities for creating and preparing data for NMF.

```@docs
generate_synthetic_data
add_gaussian_noise!
add_sparse_outliers!
normalize_nonnegative!
load_image_folder
```

---

## Visualization Functions

### Convergence Plot

Track how the algorithm converges over iterations.

```@docs
plot_convergence
```

---

### Basis Vectors

Visualize the learned basis vectors (W matrix).

```@docs
plot_basis_vectors
```

**Usage:**
- Each subplot shows one basis vector
- For images: Shows meaningful parts (e.g., facial features, object components)
- For text: Represents topics or themes

---

### Reconstruction Comparison

Compare original data vs. reconstructed data side-by-side.

```@docs
plot_reconstruction_comparison
```

---

### NMF Summary

Creates a comprehensive summary with basis vectors, reconstructions, and convergence in one figure.

```@docs
plot_nmf_summary
```

---

### Additional Visualization Functions

```@docs
plot_activation_coefficients
plot_image_reconstruction
```

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

## Performance Metrics

### RMSE (Root Mean Square Error)

Measures average reconstruction error.

```julia
rmse = sqrt(mean((X - W*H).^2))
```

- Lower is better
- Standard NMF optimizes this metric

---

### MAE (Mean Absolute Error)

Measures average absolute reconstruction error.

```julia
mae = mean(abs.(X - W*H))
```

- Lower is better
- Better metric for comparing robustness

---

### Relative Error

Error as percentage of data magnitude.

```julia
rel_error = norm(X - W*H) / norm(X)
```

- Lower is better
- Typical range: 1-20% approximately
