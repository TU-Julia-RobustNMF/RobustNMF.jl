# Non-interactive version for automated testing/benchmarking
# Usage: julia demo_robustnmf_auto.jl [1|2]
# Default: config 1 (Balanced)

using RobustNMF
using Plots
using LinearAlgebra, Statistics

# Parse command-line argument
noise_config = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1

if !(noise_config in [1, 2])
    println("Invalid config. Using default: 1")
    noise_config = 1
end

println("Running demo with noise config $noise_config ($(noise_config == 1 ? "Balanced" : "Outlier-heavy"))")

# Rest of the demo code (same as interactive version but without user input)
# Synthetic data
m, n, r = 60, 50, 5
X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=1)

if noise_config == 1
    X_noisy = copy(X)
    add_gaussian_noise!(X_noisy; sigma=0.1, clip_at_zero=true)
    add_sparse_outliers!(X_noisy; fraction=0.005, magnitude=3.0, seed=1)
elseif noise_config == 2
    X_noisy = copy(X)
    add_gaussian_noise!(X_noisy; sigma=0.02, clip_at_zero=true)
    add_sparse_outliers!(X_noisy; fraction=0.01, magnitude=8.0, seed=1)
end

# Factorizations
W_std, H_std, hist_std = nmf(X_noisy; rank=r, maxiter=200, tol=1e-5)
W_rob, H_rob, hist_rob = robust_nmf(X_noisy; rank=r, maxiter=200, tol=1e-5, seed=1)

# Compute final errors
X_rec_std = W_std * H_std
X_rec_rob = W_rob * H_rob

err_std_frob = norm(X .- X_rec_std)
err_rob_frob = norm(X .- X_rec_rob)
err_std_mae = mean(abs.(X .- X_rec_std))
err_rob_mae = mean(abs.(X .- X_rec_rob))

println("\n=== Results ===")
println("Standard NMF  - Frobenius: $(round(err_std_frob, digits=3)), MAE: $(round(err_std_mae, digits=4))")
println("Robust NMF    - Frobenius: $(round(err_rob_frob, digits=3)), MAE: $(round(err_rob_mae, digits=4))")
println("Improvement   - Frobenius: $(round((1 - err_rob_frob/err_std_frob)*100, digits=1))%, MAE: $(round((1 - err_rob_mae/err_std_mae)*100, digits=1))%")
