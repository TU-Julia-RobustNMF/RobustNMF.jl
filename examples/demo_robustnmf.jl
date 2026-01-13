using RobustNMF
using Plots
using Statistics
using LinearAlgebra

# Setting up output directory for saving plots
output_dir = "output_plots"
mkpath(output_dir)

println("="^70)
println("RobustNMF.jl Visualization Demo")
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

X_heavy_noise = copy(X_clean)
add_gaussian_noise!(X_heavy_noise; σ=0.3)
println("   • Heavy Gaussian noise (σ=0.3)")

println()

# Part 2: Run NMF on Different Datasets

println("Part 2: Running NMF")
println("-"^70)

rank = 10
maxiter = 500
tol = 1e-5

println("→ Running NMF on clean data...")
W_clean, H_clean, hist_clean = nmf(X_clean; rank=rank, maxiter=maxiter, tol=tol)
X_recon_clean = W_clean * H_clean

println("→ Running NMF on Gaussian noise data...")
W_gauss, H_gauss, hist_gauss = nmf(X_gaussian; rank=rank, maxiter=maxiter, tol=tol)
X_recon_gauss = W_gauss * H_gauss

println("→ Running NMF on outlier data...")
W_outlier, H_outlier, hist_outlier = nmf(X_outliers; rank=rank, maxiter=maxiter, tol=tol)
X_recon_outlier = W_outlier * H_outlier

println("→ Running NMF on heavy noise data...")
W_heavy, H_heavy, hist_heavy = nmf(X_heavy_noise; rank=rank, maxiter=maxiter, tol=tol)
X_recon_heavy = W_heavy * H_heavy

println()

# PART 3: Convergence Visualization

println("Part 3: Convergence Analysis")
println("-"^70)

# Plot 1: Convergence comparison
println("→ Creating convergence comparison plot...")
p_conv = plot(1:length(hist_clean), hist_clean, 
              label="Clean Data", lw=2, yscale=:log10,
              xlabel="Iteration", ylabel="Frobenius Error (log scale)",
              title="NMF Convergence: Effect of Noise",
              legend=:topright, size=(800, 500))
plot!(p_conv, 1:length(hist_gauss), hist_gauss, 
      label="Gaussian Noise (σ=0.1)", lw=2)
plot!(p_conv, 1:length(hist_outlier), hist_outlier, 
      label="Sparse Outliers (5%)", lw=2)
plot!(p_conv, 1:length(hist_heavy), hist_heavy, 
      label="Heavy Noise (σ=0.3)", lw=2)

savefig(p_conv, joinpath(output_dir, "01_convergence_comparison.png"))
println("   ✓ Saved: 01_convergence_comparison.png")

# Plot 2: Individual convergence plots
println("→ Creating individual convergence plots...")
p_conv_clean = plot_convergence(hist_clean; title="Convergence: Clean Data")
savefig(p_conv_clean, joinpath(output_dir, "02_convergence_clean.png"))

p_conv_noisy = plot_convergence(hist_gauss; title="Convergence: Noisy Data")
savefig(p_conv_noisy, joinpath(output_dir, "03_convergence_noisy.png"))

println("   ✓ Saved: 02_convergence_clean.png, 03_convergence_noisy.png")
println()


# PART 4: Basis Vectors Visualization

println("Part 4: Basis Vectors (W) Visualization")
println("-"^70)

println("→ Visualizing learned basis vectors...")

# Plot 3: Basis vectors from clean data
p_basis_clean = plot_basis_vectors(W_clean; max_components=10,
                                   title="Basis Vectors: Clean Data")
savefig(p_basis_clean, joinpath(output_dir, "04_basis_clean.png"))
println("   ✓ Saved: 04_basis_clean.png")

# Plot 4: Basis vectors from noisy data
p_basis_noisy = plot_basis_vectors(W_gauss; max_components=10,
                                   title="Basis Vectors: Gaussian Noise")
savefig(p_basis_noisy, joinpath(output_dir, "05_basis_noisy.png"))
println("   ✓ Saved: 05_basis_noisy.png")

# Plot 5: Basis vectors from outlier data
p_basis_outlier = plot_basis_vectors(W_outlier; max_components=10,
                                     title="Basis Vectors: Sparse Outliers")
savefig(p_basis_outlier, joinpath(output_dir, "06_basis_outliers.png"))
println("   ✓ Saved: 06_basis_outliers.png")

# Plot 6: Basis vectors as images (10×10)
p_basis_img = plot_basis_vectors(W_clean; img_shape=(10, 10), max_components=10,
                                title="Basis Vectors as Images")
savefig(p_basis_img, joinpath(output_dir, "07_basis_as_images.png"))
println("   ✓ Saved: 07_basis_as_images.png")
println()

# PART 5: Activation Coefficients Visualization

println("Part 5: Activation Coefficients (H) Visualization")
println("-"^70)

println("→ Visualizing activation coefficients...")

# Plot 7: Activation coefficients from clean data
p_act_clean = plot_activation_coefficients(H_clean; max_samples=8,
                                          title="Activation Coefficients: Clean Data")
savefig(p_act_clean, joinpath(output_dir, "08_activations_clean.png"))
println("   ✓ Saved: 08_activations_clean.png")

# Plot 8: Activation coefficients from noisy data
p_act_noisy = plot_activation_coefficients(H_gauss; max_samples=8,
                                          title="Activation Coefficients: Noisy Data")
savefig(p_act_noisy, joinpath(output_dir, "09_activations_noisy.png"))
println("   ✓ Saved: 09_activations_noisy.png")
println()

# Part 6: Reconstruction Visualization

println("Part 6: Data Reconstruction Visualization")
println("-"^70)

println("→ Comparing original, noisy, and reconstructed data...")

# Plot 9: Clean data reconstruction
p_recon_clean = plot_reconstruction_comparison(X_clean, X_recon_clean; 
                                              n_samples=6,
                                              title="Reconstruction: Clean Data")
savefig(p_recon_clean, joinpath(output_dir, "10_reconstruction_clean.png"))
println("   ✓ Saved: 10_reconstruction_clean.png")

# Plot 10: Noisy data reconstruction
p_recon_noisy = plot_reconstruction_comparison(X_gaussian, X_recon_gauss;
                                              n_samples=6,
                                              title="Reconstruction: Gaussian Noise")
savefig(p_recon_noisy, joinpath(output_dir, "11_reconstruction_gaussian.png"))
println("   ✓ Saved: 11_reconstruction_gaussian.png")

# Plot 11: Outlier data reconstruction
p_recon_outlier = plot_reconstruction_comparison(X_outliers, X_recon_outlier;
                                                n_samples=6,
                                                title="Reconstruction: Sparse Outliers")
savefig(p_recon_outlier, joinpath(output_dir, "12_reconstruction_outliers.png"))
println("   ✓ Saved: 12_reconstruction_outliers.png")

# Plot 12: Image reconstruction view
p_img_recon = plot_image_reconstruction(X_clean, W_clean, H_clean, (10, 10);
                                       n_images=5)
savefig(p_img_recon, joinpath(output_dir, "13_image_reconstruction.png"))
println("   ✓ Saved: 13_image_reconstruction.png")
println()

# Part 7: Summary Visualizations

println("Part 7: Comprehensive Summary Plots")
println("-"^70)

println("→ Creating summary visualizations...")

# Plot 13: Summary for clean data
p_summary_clean = plot_nmf_summary(X_clean, W_clean, H_clean, hist_clean;
                                  max_basis=9, max_samples=4)
savefig(p_summary_clean, joinpath(output_dir, "14_summary_clean.png"))
println("   ✓ Saved: 14_summary_clean.png")

# Plot 14: Summary for noisy data
p_summary_noisy = plot_nmf_summary(X_gaussian, W_gauss, H_gauss, hist_gauss;
                                  max_basis=9, max_samples=4)
savefig(p_summary_noisy, joinpath(output_dir, "15_summary_noisy.png"))
println("   ✓ Saved: 15_summary_noisy.png")
println()

# Part 8: Quantitative Evaluation

println("Part 8: Quantitative Performance Metrics")
println("-"^70)

function compute_metrics(X_orig, X_recon)
    error = X_orig - X_recon
    rmse = sqrt(mean(error.^2))
    mae = mean(abs.(error))
    rel_error = norm(error) / norm(X_orig)
    explained_var = 1 - var(vec(error)) / var(vec(X_orig))
    return (rmse=rmse, mae=mae, rel_error=rel_error, explained_var=explained_var)
end

println("\nReconstruction Quality Metrics:")
println("-"^70)

metrics_clean = compute_metrics(X_clean, X_recon_clean)
println("Clean Data:")
println("  RMSE:            $(round(metrics_clean.rmse, digits=6))")
println("  MAE:             $(round(metrics_clean.mae, digits=6))")
println("  Relative Error:  $(round(metrics_clean.rel_error*100, digits=2))%")
println("  Explained Var:   $(round(metrics_clean.explained_var*100, digits=2))%")
println()

metrics_gauss = compute_metrics(X_gaussian, X_recon_gauss)
println("Gaussian Noise (σ=0.1):")
println("  RMSE:            $(round(metrics_gauss.rmse, digits=6))")
println("  MAE:             $(round(metrics_gauss.mae, digits=6))")
println("  Relative Error:  $(round(metrics_gauss.rel_error*100, digits=2))%")
println("  Explained Var:   $(round(metrics_gauss.explained_var*100, digits=2))%")
println()

metrics_outlier = compute_metrics(X_outliers, X_recon_outlier)
println("Sparse Outliers (5%):")
println("  RMSE:            $(round(metrics_outlier.rmse, digits=6))")
println("  MAE:             $(round(metrics_outlier.mae, digits=6))")
println("  Relative Error:  $(round(metrics_outlier.rel_error*100, digits=2))%")
println("  Explained Var:   $(round(metrics_outlier.explained_var*100, digits=2))%")
println()

metrics_heavy = compute_metrics(X_heavy_noise, X_recon_heavy)
println("Heavy Noise (σ=0.3):")
println("  RMSE:            $(round(metrics_heavy.rmse, digits=6))")
println("  MAE:             $(round(metrics_heavy.mae, digits=6))")
println("  Relative Error:  $(round(metrics_heavy.rel_error*100, digits=2))%")
println("  Explained Var:   $(round(metrics_heavy.explained_var*100, digits=2))%")
println()

# Plot 15: Metrics comparison
println("→ Creating metrics comparison plot...")
metrics_names = ["Clean", "Gaussian\n(σ=0.1)", "Outliers\n(5%)", "Heavy\n(σ=0.3)"]
rmse_vals = [metrics_clean.rmse, metrics_gauss.rmse, metrics_outlier.rmse, metrics_heavy.rmse]
mae_vals = [metrics_clean.mae, metrics_gauss.mae, metrics_outlier.mae, metrics_heavy.mae]
rel_err_vals = [metrics_clean.rel_error, metrics_gauss.rel_error, 
                metrics_outlier.rel_error, metrics_heavy.rel_error] .* 100

p_metrics = plot(
    bar(metrics_names, rmse_vals, title="RMSE", legend=false, color=:steelblue),
    bar(metrics_names, mae_vals, title="MAE", legend=false, color=:coral),
    bar(metrics_names, rel_err_vals, title="Relative Error (%)", 
        legend=false, color=:seagreen),
    layout=(1, 3), size=(1200, 400),
    plot_title="Reconstruction Error Metrics"
)

savefig(p_metrics, joinpath(output_dir, "16_metrics_comparison.png"))
println("   ✓ Saved: 16_metrics_comparison.png")
println()

# Part 9: Convergence Speed Comparison

println("Part 9: Convergence Speed Analysis")
println("-"^70)

println("\nConvergence Statistics:")
println("-"^70)
println("Clean Data:         $(length(hist_clean)) iterations")
println("Gaussian Noise:     $(length(hist_gauss)) iterations")
println("Sparse Outliers:    $(length(hist_outlier)) iterations")
println("Heavy Noise:        $(length(hist_heavy)) iterations")
println()

# Plot 16: Convergence speed comparison
p_conv_speed = bar(metrics_names, 
                   [length(hist_clean), length(hist_gauss), 
                    length(hist_outlier), length(hist_heavy)],
                   xlabel="Dataset", ylabel="Iterations to Converge",
                   title="Convergence Speed Comparison",
                   legend=false, color=:purple, size=(600, 400))

savefig(p_conv_speed, joinpath(output_dir, "17_convergence_speed.png"))
println("→ Convergence speed comparison saved")
println("   ✓ Saved: 17_convergence_speed.png")
println()

# To summarize:

println("="^70)
println("Demo Complete!")
println("="^70)
