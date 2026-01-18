using RobustNMF
using Plots
using Statistics
using LinearAlgebra

# Setting up output directory for saving plots
output_dir = "output_plots_comparison"
mkpath(output_dir)

println("="^70)
println("RobustNMF.jl - Standard vs Robust NMF Comparison")
println("="^70)
println()

# Part 1: Synthetic Data Experiments

println("Part 1: Synthetic Data Experiments")
println("-"^70)

# Generating clean synthetic data
println("→ Generating synthetic data (100×60 matrix, rank 10)...")
m, n, true_rank = 100, 60, 10
X_clean, W_true, H_true = generate_synthetic_data(m, n; rank=true_rank, seed=42)

# Creating corrupted versions
println("→ Creating corrupted versions:")
X_gaussian = copy(X_clean)
add_gaussian_noise!(X_gaussian; σ=0.1)
println("   • Gaussian noise (σ=0.1)")

X_outliers = copy(X_clean)
add_sparse_outliers!(X_outliers; fraction=0.05, magnitude=5.0, seed=42)
println("   • Sparse outliers (5% corrupted, magnitude=5.0)")

X_heavy_outliers = copy(X_clean)
add_sparse_outliers!(X_heavy_outliers; fraction=0.10, magnitude=10.0, seed=123)
println("   • Heavy sparse outliers (10% corrupted, magnitude=10.0)")

println()

# Part 2: Run Different NMF Algorithms

println("Part 2: Running NMF Algorithms")
println("-"^70)

rank = 10
maxiter = 500
tol = 1e-5

# Standard NMF on different datasets
println("→ Running Standard NMF (L2)...")
println("   - On clean data...")
W_std_clean, H_std_clean, hist_std_clean = nmf(X_clean; rank=rank, maxiter=maxiter, tol=tol)

println("   - On Gaussian noise data...")
W_std_gauss, H_std_gauss, hist_std_gauss = nmf(X_gaussian; rank=rank, maxiter=maxiter, tol=tol)

println("   - On outlier data (5%)...")
W_std_outlier, H_std_outlier, hist_std_outlier = nmf(X_outliers; rank=rank, maxiter=maxiter, tol=tol)

println("   - On heavy outlier data (10%)...")
W_std_heavy, H_std_heavy, hist_std_heavy = nmf(X_heavy_outliers; rank=rank, maxiter=maxiter, tol=tol)

# Robust NMF on different datasets
println("→ Running Robust NMF (IRLS weighted)...")
println("   - On clean data...")
W_rob_clean, H_rob_clean, hist_rob_clean = robust_nmf(X_clean; rank=rank, maxiter=maxiter, tol=tol, seed=42)

println("   - On outlier data (5%)...")
W_rob_outlier, H_rob_outlier, hist_rob_outlier = robust_nmf(X_outliers; rank=rank, maxiter=maxiter, tol=tol, seed=42)

println("   - On heavy outlier data (10%)...")
W_rob_heavy, H_rob_heavy, hist_rob_heavy = robust_nmf(X_heavy_outliers; rank=rank, maxiter=maxiter, tol=tol, seed=42)

println()

# Part 3: Performance Evaluation

println("Part 3: Quantitative Performance Metrics")
println("-"^70)

function compute_metrics(X_orig, W, H)
    X_recon = W * H
    error = X_orig - X_recon
    rmse = sqrt(mean(error.^2))
    mae = mean(abs.(error))
    rel_error = norm(error) / norm(X_orig)
    explained_var = 1 - var(vec(error)) / var(vec(X_orig))
    return (rmse=rmse, mae=mae, rel_error=rel_error, explained_var=explained_var)
end

println("\n📊 Reconstruction Quality on Clean Data:")
println("-"^70)

metrics_std_clean = compute_metrics(X_clean, W_std_clean, H_std_clean)
metrics_rob_clean = compute_metrics(X_clean, W_rob_clean, H_rob_clean)

println("Standard NMF (L2):")
println("  RMSE:            $(round(metrics_std_clean.rmse, digits=6))")
println("  MAE:             $(round(metrics_std_clean.mae, digits=6))")
println("  Relative Error:  $(round(metrics_std_clean.rel_error*100, digits=2))%")
println()

println("Robust NMF (IRLS):")
println("  RMSE:            $(round(metrics_rob_clean.rmse, digits=6))")
println("  MAE:             $(round(metrics_rob_clean.mae, digits=6))")
println("  Relative Error:  $(round(metrics_rob_clean.rel_error*100, digits=2))%")
println()

println("\n📊 Performance on Outlier Data (5% outliers) - Compared to CLEAN:")
println("-"^70)

metrics_std_outlier = compute_metrics(X_clean, W_std_outlier, H_std_outlier)
metrics_rob_outlier = compute_metrics(X_clean, W_rob_outlier, H_rob_outlier)

println("Standard NMF (L2):")
println("  RMSE:            $(round(metrics_std_outlier.rmse, digits=6))")
println("  MAE:             $(round(metrics_std_outlier.mae, digits=6))")
println("  Relative Error:  $(round(metrics_std_outlier.rel_error*100, digits=2))%")
println()

println("Robust NMF (IRLS):")
println("  RMSE:            $(round(metrics_rob_outlier.rmse, digits=6))")
println("  MAE:             $(round(metrics_rob_outlier.mae, digits=6))")
println("  Relative Error:  $(round(metrics_rob_outlier.rel_error*100, digits=2))%")
println()

improvement_5pct = (metrics_std_outlier.mae - metrics_rob_outlier.mae) / metrics_std_outlier.mae * 100
println("🎯 Robust NMF improves MAE by: $(round(improvement_5pct, digits=1))%")
println()

println("\n📊 Performance on Heavy Outlier Data (10% outliers) - Compared to CLEAN:")
println("-"^70)

metrics_std_heavy = compute_metrics(X_clean, W_std_heavy, H_std_heavy)
metrics_rob_heavy = compute_metrics(X_clean, W_rob_heavy, H_rob_heavy)

println("Standard NMF (L2):")
println("  RMSE:            $(round(metrics_std_heavy.rmse, digits=6))")
println("  MAE:             $(round(metrics_std_heavy.mae, digits=6))")
println("  Relative Error:  $(round(metrics_std_heavy.rel_error*100, digits=2))%")
println()

println("Robust NMF (IRLS):")
println("  RMSE:            $(round(metrics_rob_heavy.rmse, digits=6))")
println("  MAE:             $(round(metrics_rob_heavy.mae, digits=6))")
println("  Relative Error:  $(round(metrics_rob_heavy.rel_error*100, digits=2))%")
println()

improvement_10pct = (metrics_std_heavy.mae - metrics_rob_heavy.mae) / metrics_std_heavy.mae * 100
println("🎯 Robust NMF improves MAE by: $(round(improvement_10pct, digits=1))%")
println()

# Part 4: Visualizations

println("Part 4: Creating Visualizations")
println("-"^70)

# Plot 1: Algorithm comparison on outlier data
println("→ Creating convergence comparison plot...")
p_conv = plot(title="NMF Convergence Comparison (5% Outliers)", 
              xlabel="Iteration", ylabel="Error (MAE)",
              legend=:topright, size=(900, 500), yscale=:log10)

plot!(p_conv, 1:length(hist_std_outlier), hist_std_outlier, 
      label="Standard NMF (L2)", lw=2, color=:red, linestyle=:solid)
plot!(p_conv, 1:length(hist_rob_outlier), hist_rob_outlier, 
      label="Robust NMF (IRLS)", lw=2, color=:blue, linestyle=:dash)

savefig(p_conv, joinpath(output_dir, "01_convergence_comparison.png"))
println("   ✓ Saved: 01_convergence_comparison.png")

# Plot 2: Robustness comparison bar chart
println("→ Creating robustness comparison...")
methods = ["Standard\nL2", "Robust\nIRLS"]
mae_clean = [metrics_std_clean.mae, metrics_rob_clean.mae]
mae_5pct = [metrics_std_outlier.mae, metrics_rob_outlier.mae]
mae_10pct = [metrics_std_heavy.mae, metrics_rob_heavy.mae]

p_robust = plot(
    bar(methods, mae_clean, title="MAE (Clean Data)", 
        legend=false, color=:green, ylabel="MAE"),
    bar(methods, mae_5pct, title="MAE (5% Outliers)", 
        legend=false, color=:steelblue, ylabel="MAE"),
    bar(methods, mae_10pct, title="MAE (10% Outliers)", 
        legend=false, color=:coral, ylabel="MAE"),
    layout=(1, 3), size=(1200, 400),
    plot_title="Robustness to Outliers (Lower is Better)"
)

savefig(p_robust, joinpath(output_dir, "02_robustness_comparison.png"))
println("   ✓ Saved: 02_robustness_comparison.png")

# Plot 3: Basis vectors comparison
println("→ Visualizing basis vectors...")
p_basis_std = plot_basis_vectors(W_std_outlier; max_components=9,
                                 title="Basis Vectors: Standard NMF (L2)")
savefig(p_basis_std, joinpath(output_dir, "03_basis_standard.png"))

p_basis_rob = plot_basis_vectors(W_rob_outlier; max_components=9,
                                 title="Basis Vectors: Robust NMF (IRLS)")
savefig(p_basis_rob, joinpath(output_dir, "04_basis_robust.png"))

println("   ✓ Saved basis vector plots")

# Plot 4: Reconstruction comparison
println("→ Creating reconstruction comparisons...")
X_recon_std = W_std_outlier * H_std_outlier
X_recon_rob = W_rob_outlier * H_rob_outlier

p_recon_std = plot_reconstruction_comparison(X_outliers, X_recon_std; 
                                             n_samples=6,
                                             title="Reconstruction: Standard NMF")
savefig(p_recon_std, joinpath(output_dir, "05_recon_standard.png"))

p_recon_rob = plot_reconstruction_comparison(X_outliers, X_recon_rob; 
                                             n_samples=6,
                                             title="Reconstruction: Robust NMF")
savefig(p_recon_rob, joinpath(output_dir, "06_recon_robust.png"))

println("   ✓ Saved reconstruction plots")

# Plot 5: Convergence on different corruption levels
println("→ Creating multi-condition convergence plot...")
p_multi = plot(title="Convergence Under Different Noise Conditions", 
               xlabel="Iteration", ylabel="Frobenius Error",
               legend=:topright, size=(1000, 600), yscale=:log10)

# Standard NMF
plot!(p_multi, 1:length(hist_std_clean), hist_std_clean, 
      label="Standard - Clean", lw=2, color=:green, linestyle=:solid)
plot!(p_multi, 1:length(hist_std_outlier), hist_std_outlier, 
      label="Standard - 5% Outliers", lw=2, color=:orange, linestyle=:solid)
plot!(p_multi, 1:length(hist_std_heavy), hist_std_heavy, 
      label="Standard - 10% Outliers", lw=2, color=:red, linestyle=:solid)

# Robust NMF
plot!(p_multi, 1:length(hist_rob_clean), hist_rob_clean, 
      label="Robust - Clean", lw=2, color=:lightblue, linestyle=:dash)
plot!(p_multi, 1:length(hist_rob_outlier), hist_rob_outlier, 
      label="Robust - 5% Outliers", lw=2, color=:blue, linestyle=:dash)
plot!(p_multi, 1:length(hist_rob_heavy), hist_rob_heavy, 
      label="Robust - 10% Outliers", lw=2, color=:purple, linestyle=:dash)

savefig(p_multi, joinpath(output_dir, "07_multi_convergence.png"))
println("   ✓ Saved: 07_multi_convergence.png")

println()

# Summary table
println("="^70)
println("SUMMARY TABLE - MAE on Clean Data (Lower is Better)")
println("="^70)
println("Dataset          | Standard (L2) | Robust (IRLS) | Improvement")
println("-"^70)
println("Clean            | $(round(metrics_std_clean.mae, digits=4))      | $(round(metrics_rob_clean.mae, digits=4))      | -")
println("5% Outliers      | $(round(metrics_std_outlier.mae, digits=4))      | $(round(metrics_rob_outlier.mae, digits=4))      | $(round(improvement_5pct, digits=1))%")
println("10% Outliers     | $(round(metrics_std_heavy.mae, digits=4))      | $(round(metrics_rob_heavy.mae, digits=4))      | $(round(improvement_10pct, digits=1))%")
println("="^70)

println()
println("="^70)
println("Demo Complete!")
