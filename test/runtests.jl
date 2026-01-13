using Test
using RobustNMF
@testset "RobustNMF.jl Test Suite" begin
    
    include("test_data.jl")
    
    # NMF algorithm tests
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
        @test history[end] < 1.0  # Should converge to reasonable error
    end
    
    # Visualization tests
    include("test_plotting.jl")
    
end
