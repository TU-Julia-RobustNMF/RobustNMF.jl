using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RobustNMF
using Plots
using Statistics
using LinearAlgebra
using Printf

# ------------------------------------------------------------
# Demo: Standard NMF vs Robust NMF (Huber) on synthetic data
# ------------------------------------------------------------

function run_robustnmf_demo()
    # Output directory
    output_dir = joinpath(@__DIR__, "outputs", "demo_robustnmf")
    mkpath(output_dir)

    println("="^70)
    println("RobustNMF.jl - Standard vs Robust NMF Comparison (Synthetic)")
    println("="^70)
    println()

    # ------------------------------------------------------------
    # 1) Generate data
    # ------------------------------------------------------------
    println("1. Synthetic Data Experiments")
    println("-"^70)

    m, n, true_rank = 100, 60, 10
    data_seed = 42
    init_seed = 43

    println("→ Generating synthetic data ($(m)×$(n) matrix, rank $(true_rank)) with seed=$(data_seed)...")
    X_clean, W_true, H_true = generate_synthetic_data(m, n; rank=true_rank, seed=data_seed)

    println("→ Creating corrupted versions:")
    X_gaussian = copy(X_clean)
    add_gaussian_noise!(X_gaussian; σ=0.1)
    println("   • Gaussian noise (σ=0.1)")

    X_outliers = copy(X_clean)
    add_sparse_outliers!(X_outliers; fraction=0.05, magnitude=5.0, seed=data_seed)
    println("   • Sparse outliers (5% corrupted, magnitude=5.0)")

    X_heavy_outliers = copy(X_clean)
    add_sparse_outliers!(X_heavy_outliers; fraction=0.10, magnitude=10.0, seed=123)
    println("   • Heavy sparse outliers (10% corrupted, magnitude=10.0)")
    println()

    # ------------------------------------------------------------
    # 2) Run algorithms
    # ------------------------------------------------------------
    println("2. Running NMF Algorithms")
    println("-"^70)

    rank = 10
    maxiter = 500
    tol = 1e-5
    delta = 1.0

    # Helper for consistent printing
    function run_standard(X; label)
        println("   - Standard NMF on $(label)...")
        W, H, hist = nmf(X; rank=rank, maxiter=maxiter, tol=tol, seed=init_seed)
        println("     Converged in $(length(hist)) iterations")
        return W, H, hist
    end

    function run_robust(X; label)
        println("   - Robust NMF (Huber) on $(label)...")
        W, H, hist = robustnmf(X; rank=rank, maxiter=maxiter, tol=tol, delta=delta, seed=init_seed)
        println("     Converged in $(length(hist)) iterations")
        return W, H, hist
    end

    println("→ Running Standard NMF (L2)...")
    W_std_clean,  H_std_clean,  hist_std_clean  = run_standard(X_clean;         label="clean data")
    W_std_gauss,  H_std_gauss,  hist_std_gauss  = run_standard(X_gaussian;      label="Gaussian-noise data")
    W_std_out,    H_std_out,    hist_std_out    = run_standard(X_outliers;      label="outlier data (5%)")
    W_std_heavy,  H_std_heavy,  hist_std_heavy  = run_standard(X_heavy_outliers;label="heavy outlier data (10%)")

    println("→ Running Robust NMF (Huber)...")
    W_rob_clean,  H_rob_clean,  hist_rob_clean  = run_robust(X_clean;          label="clean data")
    W_rob_gauss,  H_rob_gauss,  hist_rob_gauss  = run_robust(X_gaussian;       label="Gaussian-noise data")
    W_rob_out,    H_rob_out,    hist_rob_out    = run_robust(X_outliers;       label="outlier data (5%)")
    W_rob_heavy,  H_rob_heavy,  hist_rob_heavy  = run_robust(X_heavy_outliers; label="heavy outlier data (10%)")

    println()

    # ------------------------------------------------------------
    # 3) Metrics
    # ------------------------------------------------------------
    println("3. Quantitative Performance Metrics")
    println("-"^70)

    function compute_metrics(X_ref, W, H)
        X_hat = W * H
        E = X_ref - X_hat
        rmse = sqrt(mean(E .^ 2))
        mae = mean(abs.(E))
        rel_error = norm(E) / (norm(X_ref) + eps(eltype(X_ref)))
        return (rmse=rmse, mae=mae, rel_error=rel_error)
    end

    # Evaluate reconstructions against the CLEAN ground truth (same as your intent)
    metrics_std_clean  = compute_metrics(X_clean, W_std_clean,  H_std_clean)
    metrics_rob_clean  = compute_metrics(X_clean, W_rob_clean,  H_rob_clean)

    metrics_std_gauss  = compute_metrics(X_clean, W_std_gauss,  H_std_gauss)
    metrics_rob_gauss  = compute_metrics(X_clean, W_rob_gauss,  H_rob_gauss)

    metrics_std_out    = compute_metrics(X_clean, W_std_out,    H_std_out)
    metrics_rob_out    = compute_metrics(X_clean, W_rob_out,    H_rob_out)

    metrics_std_heavy  = compute_metrics(X_clean, W_std_heavy,  H_std_heavy)
    metrics_rob_heavy  = compute_metrics(X_clean, W_rob_heavy,  H_rob_heavy)

    function print_metrics_block(title, m_std, m_rob)
        println("\n", title)
        println("-"^70)
        println("Standard NMF (L2):")
        println("  RMSE:           ", round(m_std.rmse, digits=6))
        println("  MAE:            ", round(m_std.mae, digits=6))
        println("  Relative Error: ", round(m_std.rel_error * 100, digits=2), "%")
        println()
        println("Robust NMF (Huber):")
        println("  RMSE:           ", round(m_rob.rmse, digits=6))
        println("  MAE:            ", round(m_rob.mae, digits=6))
        println("  Relative Error: ", round(m_rob.rel_error * 100, digits=2), "%")
    end

    print_metrics_block("Reconstruction Quality (trained on CLEAN) - Compared to CLEAN:", metrics_std_clean, metrics_rob_clean)
    print_metrics_block("Performance (trained on GAUSSIAN noise) - Compared to CLEAN:",  metrics_std_gauss, metrics_rob_gauss)
    print_metrics_block("Performance (trained on 5% OUTLIERS) - Compared to CLEAN:",     metrics_std_out,   metrics_rob_out)
    print_metrics_block("Performance (trained on 10% OUTLIERS) - Compared to CLEAN:",    metrics_std_heavy,metrics_rob_heavy)

    println()

    # ------------------------------------------------------------
    # 4) Visualizations (prefer library plotting helpers)
    # ------------------------------------------------------------
    println("4. Creating Visualizations")
    println("-"^70)

    # 4.1 Convergence comparison (5% outliers)
    println("→ Creating convergence comparison plots...")
    p_conv_std = plot_convergence(hist_std_out; objective=:frobenius, log_scale=true,
                                  title="Standard NMF Convergence (5% Outliers)")
    p_conv_rob = plot_convergence(hist_rob_out; objective=:huber, log_scale=true,
                                  title="Robust NMF (Huber) Convergence (5% Outliers)")
    p_conv = plot(p_conv_std, p_conv_rob, layout=(1,2), size=(900, 400))
    savefig(p_conv, joinpath(output_dir, "01_convergence_comparison.png"))
    println("   ✓ Saved: 01_convergence_comparison.png")

    # 4.2 Robustness comparison bar chart (MAE vs CLEAN)
    println("→ Creating robustness comparison bar chart...")
    methods = ["Standard\nL2", "Robust\nHuber"]
    mae_clean  = [metrics_std_clean.mae,  metrics_rob_clean.mae]
    mae_gauss  = [metrics_std_gauss.mae,  metrics_rob_gauss.mae]
    mae_5pct   = [metrics_std_out.mae,    metrics_rob_out.mae]
    mae_10pct  = [metrics_std_heavy.mae,  metrics_rob_heavy.mae]

    p_robust = plot(
        bar(methods, mae_clean, title="MAE (Clean)",     legend=false, ylabel="MAE"),
        bar(methods, mae_gauss, title="MAE (Gaussian)",  legend=false, ylabel="MAE"),
        bar(methods, mae_5pct,  title="MAE (5% Outliers)",legend=false, ylabel="MAE"),
        bar(methods, mae_10pct, title="MAE (10% Outliers)",legend=false, ylabel="MAE"),
        layout=(1, 4), size=(1500, 400),
        plot_title="Robustness to Corruption (Lower MAE is Better)"
    )
    savefig(p_robust, joinpath(output_dir, "02_robustness_comparison.png"))
    println("   ✓ Saved: 02_robustness_comparison.png")

    # 4.3 Basis vectors comparison (5% outliers)
    println("→ Visualizing basis vectors (5% outliers case)...")
    p_basis_std = plot_basis_vectors(W_std_out; max_components=9, title="Basis Vectors: Standard NMF (5% Outliers)")
    savefig(p_basis_std, joinpath(output_dir, "03_basis_standard.png"))
    p_basis_rob = plot_basis_vectors(W_rob_out; max_components=9, title="Basis Vectors: Robust NMF (5% Outliers)")
    savefig(p_basis_rob, joinpath(output_dir, "04_basis_robust.png"))
    println("   ✓ Saved: 03_basis_standard.png, 04_basis_robust.png")

    # 4.4 Reconstruction comparison (trained on outliers, compared on outliers)
    println("→ Creating reconstruction comparisons (5% outliers case)...")
    X_recon_std = W_std_out * H_std_out
    X_recon_rob = W_rob_out * H_rob_out

    p_recon_std = plot_reconstruction_comparison(X_outliers, X_recon_std; n_samples=6,
                                                 title="Reconstruction: Standard NMF (5% Outliers)")
    savefig(p_recon_std, joinpath(output_dir, "05_recon_standard.png"))

    p_recon_rob = plot_reconstruction_comparison(X_outliers, X_recon_rob; n_samples=6,
                                                 title="Reconstruction: Robust NMF (5% Outliers)")
    savefig(p_recon_rob, joinpath(output_dir, "06_recon_robust.png"))
    println("   ✓ Saved: 05_recon_standard.png, 06_recon_robust.png")

    # 4.5 Multi-condition convergence (Standard and Robust)
    println("→ Creating multi-condition convergence plots...")

    p_std_multi = plot_convergence(hist_std_clean; objective=:frobenius, log_scale=true,
                                   title="Standard NMF - Different Corruptions")
    plot!(p_std_multi, 1:length(hist_std_gauss), hist_std_gauss, label="Gaussian")
    plot!(p_std_multi, 1:length(hist_std_out),   hist_std_out,   label="5% Outliers")
    plot!(p_std_multi, 1:length(hist_std_heavy), hist_std_heavy, label="10% Outliers")

    p_rob_multi = plot_convergence(hist_rob_clean; objective=:huber, log_scale=true,
                                   title="Robust NMF (Huber) - Different Corruptions")
    plot!(p_rob_multi, 1:length(hist_rob_gauss), hist_rob_gauss, label="Gaussian")
    plot!(p_rob_multi, 1:length(hist_rob_out),   hist_rob_out,   label="5% Outliers")
    plot!(p_rob_multi, 1:length(hist_rob_heavy), hist_rob_heavy, label="10% Outliers")

    p_multi = plot(p_std_multi, p_rob_multi, layout=(1,2), size=(1000, 500))
    savefig(p_multi, joinpath(output_dir, "07_multi_convergence.png"))
    println("   ✓ Saved: 07_multi_convergence.png")

    # 4.6 NMF summaries (outliers case)
    println("→ Creating NMF summary plots (5% outliers case)...")
    p_summary_std = plot_nmf_summary(X_outliers, W_std_out, H_std_out, hist_std_out; max_basis=9, max_samples=4)
    savefig(p_summary_std, joinpath(output_dir, "08_summary_standard.png"))

    p_summary_rob = plot_nmf_summary(X_outliers, W_rob_out, H_rob_out, hist_rob_out; max_basis=9, max_samples=4)
    savefig(p_summary_rob, joinpath(output_dir, "09_summary_robust.png"))
    println("   ✓ Saved: 08_summary_standard.png, 09_summary_robust.png")

    println()

    # ------------------------------------------------------------
    # 5) Summary table (MAE vs CLEAN)
    # ------------------------------------------------------------
    println("="^70)
    println("Summary - MAE Compared to CLEAN (Lower is Better)")
    println("="^70)
    println("Dataset          | Standard (L2) | Robust (Huber) | Improvement")
    println("-"^70)

    function improvement_str(std_mae, rob_mae)
        if std_mae <= 0 || rob_mae <= 0
            return "-"
        end
        imp = (std_mae - rob_mae) / std_mae * 100
        @sprintf("%+.1f%%", imp)
    end

    println(@sprintf("Clean            | %12.6f | %12.6f | %s",
                     metrics_std_clean.mae, metrics_rob_clean.mae,
                     improvement_str(metrics_std_clean.mae, metrics_rob_clean.mae)))

    println(@sprintf("Gaussian         | %12.6f | %12.6f | %s",
                     metrics_std_gauss.mae, metrics_rob_gauss.mae,
                     improvement_str(metrics_std_gauss.mae, metrics_rob_gauss.mae)))

    println(@sprintf("5%% Outliers      | %12.6f | %12.6f | %s",
                     metrics_std_out.mae, metrics_rob_out.mae,
                     improvement_str(metrics_std_out.mae, metrics_rob_out.mae)))

    println(@sprintf("10%% Outliers     | %12.6f | %12.6f | %s",
                     metrics_std_heavy.mae, metrics_rob_heavy.mae,
                     improvement_str(metrics_std_heavy.mae, metrics_rob_heavy.mae)))

    println("="^70)
    println()
    println("="^70)
    println("Demo is done!")
    println("="^70)

    @info "Saved outputs to $output_dir"
    return nothing
end

run_robustnmf_demo()