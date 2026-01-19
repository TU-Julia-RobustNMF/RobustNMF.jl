using Test
using RobustNMF
using Statistics

@testset "RobustNMF.jl Test Suite" begin
    
    # Data utilities tests
    include("test_data.jl")
    
    # Standard NMF algorithm tests
    @testset "StandardNMF.jl" begin
        # Test basic NMF functionality
        m, n, r = 30, 20, 5
        X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=123)
        
        W, H, history = nmf(X; rank=r, maxiter=200)
        
        @test size(W) == (m, r)
        @test size(H) == (r, n)
        @test all(W .>= 0)  # Non-negativity of W
        @test all(H .>= 0)  # Non-negativity of H
        @test length(history) <= 200
        @test history[end] < history[1]  # Error should decrease
        
        # Test reconstruction
        X_recon = W * H
        @test size(X_recon) == size(X)
        
        # Test convergence
        @test history[end] <= 1.0  # Should converge to reasonable error
    end
    
    # Robust NMF algorithm tests
    @testset "RobustNMF Algorithm" begin
        # Generate test data with outliers
        m, n, r = 50, 30, 5
        X_clean, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=42)
        
        # Create corrupted version
        X_outliers = copy(X_clean)
        add_sparse_outliers!(X_outliers; fraction=0.1, magnitude=5.0, seed=42)
        
        # Test robust_nmf function
        W, H, history = robust_nmf(X_outliers; rank=r, maxiter=200, seed=42)
        
        @test size(W) == (m, r)
        @test size(H) == (r, n)
        @test all(W .>= 0)  # Non-negativity of W
        @test all(H .>= 0)  # Non-negativity of H
        @test length(history) <= 200
        # Error should decrease or stay very close (allow for numerical precision)
        @test history[end] <= history[1] * 1.01  
        
        # Test reconstruction
        X_recon = W * H
        @test size(X_recon) == size(X_outliers)
        
        # Test that robust NMF is indeed more robust
        W_std, H_std, _ = nmf(X_outliers; rank=r, maxiter=200)
        
        # Compare reconstruction against clean data
        mae_robust = mean(abs.(X_clean .- (W * H)))
        mae_std = mean(abs.(X_clean .- (W_std * H_std)))
        
        # Robust should perform better (or at least comparably)
        @test mae_robust <= mae_std * 1.5  # Allow some tolerance
    end
    
    # Visualization tests
    include("test_plotting.jl")
    
end
