using RobustNMF
using Plots

# Example 1: Synthetic data
println("Testing with synthetic data...")
X, W_true, H_true = generate_synthetic_data(100, 50; rank=10, seed=42)
add_gaussian_noise!(X; σ=0.1)

W, H, history = nmf(X; rank=10, maxiter=300)

# Create visualizations
p1 = plot_convergence(history)
savefig(p1, "convergence.png")

p2 = plot_basis_vectors(W; max_components=10)
savefig(p2, "basis_vectors.png")

p3 = plot_nmf_summary(X, W, H, history)
savefig(p3, "nmf_summary.png")

println("Plots saved!")
