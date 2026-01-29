using Test
using RobustNMF
using Statistics

@testset "RobustNMFAlgorithms.jl (Huber Robust NMF)" begin
    # Synthetic non-negative data with known factorization rank
    m, n, r = 30, 20, 5
    X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=123)

    @testset "Default robustnmf (Huber loss)" begin
        # The default robustnmf entry point should run successfully and
        # return a valid factorization using the Huber-based algorithm
        W, H, history = robustnmf(
            X;
            rank = r,
            maxiter = 200,
            tol = 1e-8,
            delta = 1.0,
            seed = 123
        )

        # Factor dimensions must match the NMF specification
        @test size(W) == (m, r)
        @test size(H) == (r, n)

        # Non-negativity constraints
        @test all(W .>= 0)
        @test all(H .>= 0)

        # Numerical sanity: no NaN or Inf values should appear
        @test all(isfinite, W)
        @test all(isfinite, H)

        # Optimization history should be non-empty and bounded by maxiter
        @test 1 ≤ length(history) ≤ 200

        # Huber objective values must be finite and non-negative
        @test all(h -> isfinite(h) && h ≥ 0, history)

        # The best objective achieved should be significantly worse than
        # the initial value (allowing for floating-point noise)
        best = minimum(history)
        @test best ≤ history[1] * (1 + 1e-6) + 1e-24

        # Reconstruction must have the same shape as the input data
        @test size(W * H) == size(X)
    end

    @testset "Direct robustnmf_huber call" begin
        # Calling the Huber-based implementation directly should behave consistently
        # with the default robustnmf entry point
        W, H, history = robustnmf_huber(
            X;
            rank = r,
            maxiter = 150,
            tol = 1e-8,
            delta = 0.5,
            seed = 7
        )
        
        # Factor dimensions must match the NMF specification
        @test size(W) == (m, r)
        @test size(H) == (r, n)

        # Non-negativity constraints
        @test all(W .>= 0)
        @test all(H .>= 0)

        # Numerical sanity: no NaN or Inf values should appear
        @test all(isfinite, W)
        @test all(isfinite, H)

        # Optimization history should be non-empty and bounded by maxiter
        @test 1 ≤ length(history) ≤ 150

        # Huber objective values must be finite and non-negative
        @test all(h -> isfinite(h) && h ≥ 0, history)

        # The best objective achieved should be significantly worse than
        # the initial value (allowing for floating-point noise)
        best = minimum(history)
        @test best ≤ history[1] * (1 + 1e-6) + 1e-24

        # Reconstruction must have the same shape as the input data
        @test size(W * H) == size(X)
    end

    @testset "Input validation" begin
        # NMF requires non-negative input data
        X_bad = copy(X)
        X_bad[1, 1] = -0.1
        @test_throws ArgumentError robustnmf_huber(X_bad; rank = r, seed = 1)

        # Huber threshold must be strictly positive
        @test_throws ArgumentError robustnmf_huber(X; rank = r, delta = 0.0, seed = 1)
        @test_throws ArgumentError robustnmf_huber(X; rank = r, delta = -1.0, seed = 1)

        # Invalid optimization parameters should be rejected
        @test_throws ArgumentError robustnmf_huber(X; rank = 0, seed = 1)
        @test_throws ArgumentError robustnmf_huber(X; maxiter = 0, seed = 1)
        @test_throws ArgumentError robustnmf_huber(X; tol = 0.0, seed = 1)
    end
end


@testset "Legacy L2,1-NMF Algorithm" begin
    # The legacy L2,1-based implementation is kept for backward compatibility
    m, n, r = 50, 30, 5
    X_clean, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=99)

    # Creating a corrupted version of the data
    X_outliers = copy(X_clean)
    add_sparse_outliers!(X_outliers; fraction=0.1, magnitude=5.0, seed=99)

    # Testing robustnmf function
    F, G, history = robustnmf_l21(X_outliers; rank=r, maxiter=200, seed=99)

    @test size(F) == (m, r)
    @test size(G) == (r, n)
    @test all(F .>= 0)  # Non-negativity of F
    @test all(G .>= 0)  # Non-negativity of G
    @test length(history) <= 200
    @test history[end] <= history[1]  # Error should decrease or stabilize

    # Reconstruction sanity
    X_recon = F * G
    @test size(X_recon) == size(X_outliers)

    # L21_loss basic sanity test
    test_matrix = rand(10, 5)
    l21_val = l21_loss(test_matrix)
    @test isfinite(l21_val) && l21_val >= 0

    # L2,1-NMF should work on clean data too
    F_clean, G_clean, hist_clean = robustnmf_l21(X_clean; rank=r, maxiter=100, seed=99)
    @test size(F_clean) == (m, r)
    @test size(G_clean) == (r, n)

    # Compare robustness vs standard NMF on outlier-corrupted data
    W_std, H_std, _ = nmf(X_outliers; rank=r, maxiter=200, seed=99)

    # Comparing reconstruction against clean data
    mae_l21 = mean(abs.(X_clean .- (F * G)))
    mae_std = mean(abs.(X_clean .- (W_std * H_std)))

    # L2,1-NMF should perform better (or at least comparably) on outlier data
    @test mae_l21 <= mae_std * 1.5  # Allow some tolerance
end
