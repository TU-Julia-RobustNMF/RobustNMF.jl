# Examples and Use Cases

Practical workflows demonstrating RobustNMF.jl capabilities.

---

## Example 1: Generate Data with Outliers

Create synthetic data and corrupt it with outliers to see how robust NMF handles them.

```julia
using RobustNMF, Statistics

# Generate clean synthetic data
X_clean, W_true, H_true = generate_synthetic_data(100, 60; rank=10, seed=42)

# Create corrupted version with outliers
X_outliers = copy(X_clean)
add_sparse_outliers!(X_outliers; fraction=0.05, magnitude=5.0, seed=42)
```

---

## Example 2: Train Both Algorithms on Noisy Data

Compare standard NMF and robust NMF on the same corrupted dataset.

```julia
# Train standard NMF on data with outliers
W_std, H_std, hist_std = nmf(X_outliers; rank=10, maxiter=500, tol=1e-5)

# Train robust NMF (Huber loss) on the same data
W_rob, H_rob, hist_rob = robustnmf(X_outliers; rank=10, maxiter=500, delta=1.0, seed=42)
```

---

## Example 3: Compare Performance on Clean Data

Evaluate both algorithms on the original clean data to see which generalizes better.

```julia
# Evaluate reconstruction error on original clean data
mae_standard = mean(abs.(X_clean - W_std*H_std))
mae_robust = mean(abs.(X_clean - W_rob*H_rob))

println("Standard NMF MAE: $mae_standard")
println("Robust NMF MAE:   $mae_robust")
println("Improvement:      $(round((mae_standard - mae_robust)/mae_standard*100, digits=1))%")
```

---

## Example 4: Visualize Results

Generate comprehensive visualizations to understand the results.

```julia
using Plots

# Full summary for Standard NMF
plot_nmf_summary(X_outliers, W_std, H_std, hist_std; title="Standard NMF")

# Full summary for Robust NMF
plot_nmf_summary(X_outliers, W_rob, H_rob, hist_rob; title="Robust NMF")

# Individual basis vectors
plot_basis_vectors(W_std; max_components=9, title="Standard NMF Basis")
plot_basis_vectors(W_rob; max_components=9, title="Robust NMF Basis")

# Compare convergence
plot_convergence(hist_std; objective=:frobenius, title="Standard NMF Convergence")
plot_convergence(hist_rob; objective=:huber, title="Robust NMF Convergence")
```

---

## Running the Full Demo

Run the comprehensive comparison demo to see both algorithms in action:

```julia
include("examples/demo_robustnmf.jl")
```

This generates plots comparing Standard NMF vs. Robust NMF on multiple datasets with varying outlier levels.
