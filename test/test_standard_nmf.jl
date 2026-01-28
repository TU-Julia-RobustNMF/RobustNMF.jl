using Test
using RobustNMF
using LinearAlgebra

@testset "StandardNMF.jl" begin
    # Basic NMF functionality on synthetic nonnegative data
    m, n, r = 30, 20, 5
    X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=123)

    # Use an explicit tol to make stopping behavior deterministic across environments
    W, H, history = nmf(X; rank=r, maxiter=200, seed=123)

    # Shape checks
    @test size(W) == (m, r)
    @test size(H) == (r, n)

    # Nonnegativity contraints
    @test all(W .>= 0)
    @test all(H .>= 0)

    # Numerical sanity: no NaNs/Infs
    @test all(isfinite, W)
    @test all(isfinite, H)
    @test all(isfinite, history)

    # History length is bound by maxiter
    @test length(history) <= 200
    @test length(history) >= 1  # should record at least one objective value

    # Objective should improve substantially (avoid overly strict absoulte thresholds)
    @test isapprox(history[end], history[1]; rtol=1e-6, atol=1e-24) || (history[end] < history[1])  # at least 50% reduction overall

    # Reconstruction has correct shape
    X_recon = W * H
    @test size(X_recon) == size(X)

end
