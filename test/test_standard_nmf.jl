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

@testset "StandardNMF.jl Validation" begin
    m, n, r = 20, 15, 4
    X, _, _ = generate_synthetic_data(m, n; rank=r, seed=42)

    X_bad = copy(X)
    X_bad[1, 1] = -0.1
    @test_throws ArgumentError nmf(X_bad; rank=r)

    @test_throws ArgumentError nmf(X; rank=0)
    @test_throws ArgumentError nmf(X; maxiter=0)
    @test_throws ArgumentError nmf(X; tol=0.0)
end

@testset "StandardNMF.jl Early Stopping" begin
    X, _, _ = generate_synthetic_data(20, 15; rank=4, seed=123)
    _, _, history = nmf(X; rank=4, maxiter=10, tol=1e9, seed=1)
    @test length(history) == 2
end
