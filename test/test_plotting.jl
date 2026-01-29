using Test
using RobustNMF
using Plots
using Statistics

@testset "Plotting Functions" begin
    
    # Generate test data
    m, n, r = 50, 30, 5
    X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=42)
    add_gaussian_noise!(X; σ=0.05)
    
    # Run NMF to get factors
    W, H, history = nmf(X; rank=r, maxiter=100)
    X_recon = W * H
    
    @testset "plot_convergence" begin
        p = plot_convergence(history)
        @test p isa Plots.Plot
        @test length(p.series_list) == 1  # Should have one line
        
        # Test with linear scale
        p_linear = plot_convergence(history; log_scale=false)
        @test p_linear isa Plots.Plot
    end
    
    @testset "plot_basis_vectors" begin
        # Test without image shape (1D visualization)
        p = plot_basis_vectors(W; max_components=5)
        @test p isa Plots.Plot
        
        # Test with image shape
        m_img = 100
        X_img, _, _ = generate_synthetic_data(m_img, 20; rank=5, seed=123)
        W_img, _, _ = nmf(X_img; rank=5, maxiter=50)
        
        p_img = plot_basis_vectors(W_img; img_shape=(10, 10), max_components=5)
        @test p_img isa Plots.Plot
    end
    
    @testset "plot_activation_coefficients" begin
        p = plot_activation_coefficients(H; max_samples=5)
        @test p isa Plots.Plot
        
        # Test with small matrix (should show heatmap)
        H_small = rand(5, 10)
        p_heatmap = plot_activation_coefficients(H_small)
        @test p_heatmap isa Plots.Plot
    end
    
    @testset "plot_reconstruction_comparison" begin
        # Test without image shape
        p = plot_reconstruction_comparison(X, X_recon; n_samples=3)
        @test p isa Plots.Plot
        
        # Test with image shape
        m_img = 64
        X_img, _, _ = generate_synthetic_data(m_img, 10; rank=4, seed=456)
        W_img, H_img, _ = nmf(X_img; rank=4, maxiter=50)
        X_recon_img = W_img * H_img
        
        p_img = plot_reconstruction_comparison(X_img, X_recon_img; 
                                               img_shape=(8, 8), n_samples=3)
        @test p_img isa Plots.Plot
    end
    
    @testset "plot_nmf_summary" begin
        p = plot_nmf_summary(X, W, H, history; max_basis=4, max_samples=2)
        @test p isa Plots.Plot
        
        # Test with image shape
        m_img = 64
        X_img, _, _ = generate_synthetic_data(m_img, 15; rank=6, seed=789)
        W_img, H_img, hist_img = nmf(X_img; rank=6, maxiter=80)
        
        p_img = plot_nmf_summary(X_img, W_img, H_img, hist_img; 
                                img_shape=(8, 8), max_basis=6, max_samples=3)
        @test p_img isa Plots.Plot
    end
    
    @testset "plot_image_reconstruction" begin
        # Generate image-like data
        m_img = 100  # 10x10 images
        X_img, _, _ = generate_synthetic_data(m_img, 8; rank=5, seed=999)
        W_img, H_img, _ = nmf(X_img; rank=5, maxiter=50)
        
        p = plot_image_reconstruction(X_img, W_img, H_img, (10, 10); n_images=3)
        @test p isa Plots.Plot
        
        # Test with specific indices
        p_idx = plot_image_reconstruction(X_img, W_img, H_img, (10, 10); 
                                         indices=[1, 2, 3], n_images=3)
        @test p_idx isa Plots.Plot
    end
    
end


@testset "Visualization Demo - Save Plots" begin
    
    println("\n" * "="^60)
    println("Running Visualization Demo")
    println("="^60)
    
    # Create output directory for plots in a writable temp location
    output_dir = mktempdir()
    
    # Generate synthetic data with noise
    println("\n1. Generating synthetic data...")
    m, n, r = 100, 60, 10
    X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=42)
    
    # Add different types of noise
    X_gaussian = copy(X)
    add_gaussian_noise!(X_gaussian; σ=0.1)
    
    X_outliers = copy(X)
    add_sparse_outliers!(X_outliers; fraction=0.05, magnitude=3.0, seed=42)
    
    # Run NMF on clean and noisy data
    println("2. Running NMF on clean data...")
    W_clean, H_clean, hist_clean = nmf(X; rank=r, maxiter=300, tol=1e-5)
    
    println("3. Running NMF on noisy data (Gaussian)...")
    W_gauss, H_gauss, hist_gauss = nmf(X_gaussian; rank=r, maxiter=300, tol=1e-5)
    
    println("4. Running NMF on noisy data (outliers)...")
    W_outlier, H_outlier, hist_outlier = nmf(X_outliers; rank=r, maxiter=300, tol=1e-5)
    
    # Create and save plots
    println("\n5. Creating visualizations...")
    
    # Convergence comparison
    p1 = plot(1:length(hist_clean), hist_clean, 
             label="Clean Data", lw=2, yscale=:log10,
             xlabel="Iteration", ylabel="Frobenius Error",
             title="Convergence Comparison", legend=:topright)
    plot!(p1, 1:length(hist_gauss), hist_gauss, 
          label="Gaussian Noise", lw=2)
    plot!(p1, 1:length(hist_outlier), hist_outlier, 
          label="Sparse Outliers", lw=2)
    savefig(p1, joinpath(output_dir, "convergence_comparison.png"))
    println("   ✓ Saved: convergence_comparison.png")
    
    # Basis vectors
    p2 = plot_basis_vectors(W_clean; max_components=10, 
                           title="Basis Vectors (Clean Data)")
    savefig(p2, joinpath(output_dir, "basis_vectors_clean.png"))
    println("   ✓ Saved: basis_vectors_clean.png")
    
    p3 = plot_basis_vectors(W_gauss; max_components=10,
                           title="Basis Vectors (Gaussian Noise)")
    savefig(p3, joinpath(output_dir, "basis_vectors_noisy.png"))
    println("   ✓ Saved: basis_vectors_noisy.png")
    
    # Activation coefficients
    p4 = plot_activation_coefficients(H_clean; max_samples=8,
                                     title="Activation Coefficients")
    savefig(p4, joinpath(output_dir, "activations.png"))
    println("   ✓ Saved: activations.png")
    
    # Reconstruction comparison
    X_recon_clean = W_clean * H_clean
    p5 = plot_reconstruction_comparison(X, X_recon_clean; n_samples=5,
                                       title="Reconstruction (Clean)")
    savefig(p5, joinpath(output_dir, "reconstruction_clean.png"))
    println("   ✓ Saved: reconstruction_clean.png")
    
    X_recon_gauss = W_gauss * H_gauss
    p6 = plot_reconstruction_comparison(X_gaussian, X_recon_gauss; n_samples=5,
                                       title="Reconstruction (Noisy)")
    savefig(p6, joinpath(output_dir, "reconstruction_noisy.png"))
    println("   ✓ Saved: reconstruction_noisy.png")
    
    # Summary plot
    p7 = plot_nmf_summary(X, W_clean, H_clean, hist_clean; 
                         max_basis=9, max_samples=4)
    savefig(p7, joinpath(output_dir, "nmf_summary.png"))
    println("   ✓ Saved: nmf_summary.png")
    
    # Image reconstruction (treating data as 10x10 images)
    if m == 100
        p8 = plot_image_reconstruction(X, W_clean, H_clean, (10, 10); n_images=5)
        savefig(p8, joinpath(output_dir, "image_reconstruction.png"))
        println("   ✓ Saved: image_reconstruction.png")
    end
    
    println("\n" * "="^60)
    println("All plots saved to: $output_dir")
    println("="^60 * "\n")
    
    @test isdir(output_dir)
    @test isfile(joinpath(output_dir, "convergence_comparison.png"))
    @test isfile(joinpath(output_dir, "nmf_summary.png"))
    
end
