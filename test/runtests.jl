using Test
using RobustNMF
using Statistics  # Add this import for mean()
using LinearAlgebra

@testset verbose = true "RobustNMF.jl Test Suite" begin
    
    # Data utilities tests
    @testset "Data Utilities" begin
        include("test_data.jl")
    end
    
    # Standard NMF algorithm tests
    @testset "StandardNMF.jl" begin
        # Testing basic NMF functionality
        m, n, r = 30, 20, 5
        X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=123)
        
        W, H, history = nmf(X; rank=r, maxiter=200, seed=123)
        
        @test size(W) == (m, r)
        @test size(H) == (r, n)
        @test all(W .>= 0)  # Non-negativity of W
        @test all(H .>= 0)  # Non-negativity of H
        @test length(history) <= 200
        @test history[end] <= history[1] + 1e-12 # Changed from < to <= to handle near-convergence
        
        # Testing reconstruction
        X_recon = W * H
        @test size(X_recon) == size(X)
        
        # Testing convergence
        @test history[end] / norm(X) < 1e-10  # Should converge to reasonable error
    end
   
    # L2,1-NMF algorithm tests
    @testset "Robust NMF (L2,1)" begin
        include("test_robust_nmf.jl")
    end
    
    # Visualization tests
    @testset "Plotting" begin
        include("test_plotting.jl")
    end
    
end
