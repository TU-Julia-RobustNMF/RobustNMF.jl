using RobustNMF
using LinearAlgebra, Statistics

# Synthetic data
m, n, r = 60, 50, 5
X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=1)

# Corrupt data (favor robust NMF with sparse, strong outliers)
X_noisy = copy(X)
add_gaussian_noise!(X_noisy; σ=0.02, clip_at_zero=true)
add_sparse_outliers!(X_noisy; fraction=0.01, magnitude=8.0, seed=1)

# Factorizations
W_std, H_std, hist_std = nmf(X_noisy; rank=r, maxiter=200, tol=1e-5)
W_rob, H_rob, hist_rob = robust_nmf(X_noisy; rank=r, maxiter=200, tol=1e-5, seed=1)

# Compare against clean X
err_std = norm(X .- W_std * H_std)
err_rob = norm(X .- W_rob * H_rob)
l1_std = mean(abs.(X .- W_std * H_std))
l1_rob = mean(abs.(X .- W_rob * H_rob))
println("Frobenius vs clean X: std=$err_std, robust=$err_rob")
println("L1 vs clean X: std=$l1_std, robust=$l1_rob")

# Plot output directory
plot_dir = "plots"
isdir(plot_dir) || mkpath(plot_dir)

# Factor plots
plot_factors(W_std, H_std; savepath=joinpath(plot_dir, "factors_standard.png"), display_plot=true)
plot_factors(W_rob, H_rob; savepath=joinpath(plot_dir, "factors_robust.png"), display_plot=true)

# Reconstruction plots
plot_reconstruction(X, X_noisy, W_std, H_std;
    savepath=joinpath(plot_dir, "reconstruction_standard.png"),
    display_plot=true,
    shared_clims=true
)
plot_reconstruction(X, X_noisy, W_rob, H_rob;
    savepath=joinpath(plot_dir, "reconstruction_robust.png"),
    display_plot=true,
    shared_clims=true
)

# Convergence comparison (Frobenius)
plot_convergence([hist_std, hist_rob];
    labels=["standard nmf", "robust nmf"],
    savepath=joinpath(plot_dir, "convergence.png"),
    display_plot=true
)
