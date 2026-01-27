using Test
using RobustNMF
using Statistics

@testset "L2,1-NMF Algorithm" begin
    # Generating test data with outliers
    m, n, r = 50, 30, 5
    X_clean, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=99)

    # Creating corrupted version
    X_outliers = copy(X_clean)
    add_sparse_outliers!(X_outliers; fraction=0.1, magnitude=5.0, seed=99)

    # Testing robustnmf function
    F, G, history = robustnmf(X_outliers; rank=r, maxiter=200, seed=99)

    @test size(F) == (m, r)
    @test size(G) == (r, n)
    @test all(F .>= 0)  # Non-negativity of F
    @test all(G .>= 0)  # Non-negativity of G
    @test length(history) <= 200
    @test history[end] <= history[1]  # Error should decrease or stabilize

    # Testing reconstruction
    X_recon = F * G
    @test size(X_recon) == size(X_outliers)

    # QUESTION: function not used? just in example in RobustNMFAlgorithms
    # Testing l21norm function
    test_matrix = rand(10, 5)
    l21_val = l21norm(test_matrix)
    @test l21_val >= 0
    @test typeof(l21_val) == Float64

    # Testing that L2,1-NMF works on clean data too
    F_clean, G_clean, hist_clean = robustnmf(X_clean; rank=r, maxiter=100, seed=99)
    @test size(F_clean) == (m, r)
    @test size(G_clean) == (r, n)

    # Testing that L2,1-NMF is more robust than standard NMF
    F_std, G_std, _ = nmf(X_outliers; rank=r, maxiter=200)

    # Comparing reconstruction against clean data
    mae_l21 = mean(abs.(X_clean .- (F * G)))
    mae_std = mean(abs.(X_clean .- (F_std * G_std)))

    # L2,1-NMF should perform better (or at least comparably) on outlier data
    @test mae_l21 <= mae_std * 1.5  # Allow some tolerance
end
