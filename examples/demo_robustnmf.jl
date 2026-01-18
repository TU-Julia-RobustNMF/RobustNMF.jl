using RobustNMF
using Plots
using LinearAlgebra, Statistics

# Synthetic data
m, n, r = 60, 50, 5
X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=1)

# Corrupt data (favor robust NMF with sparse, strong outliers)
X_noisy = copy(X)
add_gaussian_noise!(X_noisy; sigma=0.02, clip_at_zero=true)
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

# Factor plots (individual scaling)
plot_factors(W_std, H_std; savepath=joinpath(plot_dir, "factors_standard.png"), display_plot=true)
plot_factors(W_rob, H_rob; savepath=joinpath(plot_dir, "factors_robust.png"), display_plot=true)

# Factor plots (shared scaling for fair comparison)
clims_W = extrema(vcat(vec(W_std), vec(W_rob)))
clims_H = extrema(vcat(vec(H_std), vec(H_rob)))
plot_factors(W_std, H_std;
    clims_W=clims_W,
    clims_H=clims_H,
    savepath=joinpath(plot_dir, "factors_standard_shared.png"),
    display_plot=true
)
plot_factors(W_rob, H_rob;
    clims_W=clims_W,
    clims_H=clims_H,
    savepath=joinpath(plot_dir, "factors_robust_shared.png"),
    display_plot=true
)

# # Reconstruction plots
# plot_reconstruction(X, X_noisy, W_std, H_std;
#     savepath=joinpath(plot_dir, "reconstruction_standard.png"),
#     display_plot=true,
#     shared_clims=true
# )
# plot_reconstruction(X, X_noisy, W_rob, H_rob;
#     savepath=joinpath(plot_dir, "reconstruction_robust.png"),
#     display_plot=true,
#     shared_clims=true
# )

# Robustness comparison (shared clims + residuals)
X_std = W_std * H_std
X_rob = W_rob * H_rob
res_std = abs.(X .- X_std)
res_rob = abs.(X .- X_rob)

clims_data = extrema(vcat(vec(X), vec(X_noisy), vec(X_std), vec(X_rob)))
clims_res = extrema(vcat(vec(res_std), vec(res_rob)))

p11 = heatmap(X; title="X", clims=clims_data, colorbar=false)
p12 = heatmap(X_noisy; title="X_noisy", clims=clims_data, colorbar=false)
p13 = heatmap(X_std; title="W*H (std)", clims=clims_data, colorbar=false)
p14 = heatmap(res_std; title="|X - W*H| (std)", clims=clims_res, colorbar=true)

p21 = heatmap(X; title="X", clims=clims_data, colorbar=false)
p22 = heatmap(X_noisy; title="X_noisy", clims=clims_data, colorbar=false)
p23 = heatmap(X_rob; title="W*H (robust)", clims=clims_data, colorbar=false)
p24 = heatmap(res_rob; title="|X - W*H| (robust)", clims=clims_res, colorbar=true)

p = plot(p11, p12, p13, p14, p21, p22, p23, p24; layout=(2, 4), size=(1600, 800))
savefig(p, joinpath(plot_dir, "robustness_comparison.png"))

# Convergence (separate plots with their native metrics)
plot_convergence(hist_std;
    label="standard nmf (Frobenius)",
    savepath=joinpath(plot_dir, "convergence_standard.png"),
    display_plot=true
)
plot_convergence(hist_rob;
    label="robust nmf (MAE)",
    savepath=joinpath(plot_dir, "convergence_robust.png"),
    display_plot=true
)
