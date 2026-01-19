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

# Compute final errors against clean X (post-hoc metrics)
X_rec_std = W_std * H_std
X_rec_rob = W_rob * H_rob

err_std_frob = norm(X .- X_rec_std)
err_rob_frob = norm(X .- X_rec_rob)
err_std_mae = mean(abs.(X .- X_rec_std))
err_rob_mae = mean(abs.(X .- X_rec_rob))

println("=== Final Reconstruction Errors vs. Clean X ===")
println("Standard NMF  - Frobenius: $(round(err_std_frob, digits=3)), MAE: $(round(err_std_mae, digits=4))")
println("Robust NMF    - Frobenius: $(round(err_rob_frob, digits=3)), MAE: $(round(err_rob_mae, digits=4))")
println("Improvement   - Frobenius: $(round((1 - err_rob_frob/err_std_frob)*100, digits=1))%, MAE: $(round((1 - err_rob_mae/err_std_mae)*100, digits=1))%")

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
res_std = abs.(X .- X_rec_std)
res_rob = abs.(X .- X_rec_rob)

clims_data = extrema(vcat(vec(X), vec(X_noisy), vec(X_rec_std), vec(X_rec_rob)))
clims_res = extrema(vcat(vec(res_std), vec(res_rob)))

p11 = heatmap(X; title="X (clean)", clims=clims_data, colorbar=false)
p12 = heatmap(X_noisy; title="X_noisy", clims=clims_data, colorbar=false)
p13 = heatmap(X_rec_std; title="W*H (std)\nFrob: $(round(err_std_frob, digits=2))",
              clims=clims_data, colorbar=false, titlefontsize=9)
p14 = heatmap(res_std; title="|X - W*H| (std)\nMAE: $(round(err_std_mae, digits=3))",
              clims=clims_res, colorbar=true, titlefontsize=9)

p21 = heatmap(X; title="X (clean)", clims=clims_data, colorbar=false)
p22 = heatmap(X_noisy; title="X_noisy", clims=clims_data, colorbar=false)
p23 = heatmap(X_rec_rob; title="W*H (robust)\nFrob: $(round(err_rob_frob, digits=2))",
              clims=clims_data, colorbar=false, titlefontsize=9)
p24 = heatmap(res_rob; title="|X - W*H| (robust)\nMAE: $(round(err_rob_mae, digits=3))",
              clims=clims_res, colorbar=true, titlefontsize=9)

p = plot(p11, p12, p13, p14, p21, p22, p23, p24; layout=(2, 4), size=(1600, 800))
savefig(p, joinpath(plot_dir, "robustness_comparison.png"))

# Convergence plots (separate, showing native training metrics)
p_conv_std = plot(hist_std;
    label="Standard NMF",
    xlabel="Iteration",
    ylabel="Frobenius Error (training on X_noisy)",
    title="Standard NMF: Training Convergence\n" *
          "Final errors vs. clean X: Frob=$(round(err_std_frob, digits=2)), MAE=$(round(err_std_mae, digits=3))",
    legend=:topright,
    linewidth=2,
    titlefontsize=10,
    size=(800, 500)
)
savefig(p_conv_std, joinpath(plot_dir, "convergence_standard.png"))
display(p_conv_std)

p_conv_rob = plot(hist_rob;
    label="Robust NMF",
    xlabel="Iteration",
    ylabel="Mean Absolute Error (training on X_noisy)",
    title="Robust NMF: Training Convergence\n" *
          "Final errors vs. clean X: Frob=$(round(err_rob_frob, digits=2)), MAE=$(round(err_rob_mae, digits=3))",
    legend=:topright,
    linewidth=2,
    color=:red,
    titlefontsize=10,
    size=(800, 500)
)
savefig(p_conv_rob, joinpath(plot_dir, "convergence_robust.png"))
display(p_conv_rob)

# Final error comparison (Bar chart with comparable metrics)
using StatsPlots

metric_names = ["Frobenius\nError", "Mean Absolute\nError"]
std_errors = [err_std_frob, err_std_mae]
rob_errors = [err_rob_frob, err_rob_mae]

p_metrics = groupedbar(
    [std_errors rob_errors];
    bar_position=:dodge,
    labels=["Standard NMF" "Robust NMF"],
    xlabel="Metric",
    ylabel="Reconstruction Error (vs. Clean X)",
    title="Final Reconstruction Quality Comparison",
    xticks=(1:2, metric_names),
    legend=:topright,
    size=(700, 500),
    color=[:steelblue :coral],
    titlefontsize=11
)
savefig(p_metrics, joinpath(plot_dir, "metrics_comparison.png"))
display(p_metrics)
