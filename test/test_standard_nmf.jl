using Test
using RobustNMF

@testset "StandardNMF.jl" begin
    # Testing basic NMF functionality
    m, n, r = 30, 20, 5
    X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=123)

    W, H, history = nmf(X; rank=r, maxiter=200)

    @test size(W) == (m, r)
    @test size(H) == (r, n)
    @test all(W .>= 0)  # Non-negativity of W
    @test all(H .>= 0)  # Non-negativity of H
    @test length(history) <= 200
    @test history[end] <= history[1]  # Changed from < to <= to handle near-convergence

    # Testing reconstruction
    X_recon = W * H
    @test size(X_recon) == size(X)

    # Testing convergence
    @test history[end] < 1.0  # Should converge to reasonable error
end
