using Test
using RobustNMF
using FileIO: save
using ColorTypes: Gray

@testset "generate_synthetic_data" begin
    
    m, n, r = 20, 10, 5

    X, W, H = generate_synthetic_data(m, n; rank=r, noise_level=0.1, seed=123)

    @test size(W) == (m, r)
    @test size(H) == (r, n)
    @test size(X) == (m, n)
    
    # X should be non-negative after clipping
    @test all(X .>= 0)

    # With zero noise we should get exactly W*H
    X2, W2, H2 = generate_synthetic_data(m, n; rank=r, noise_level=0.0, seed=123)
    @test X2 == W2 * H2
    @test all(X2 .>= 0)

end


@testset "add_gaussian_noise!" begin
    
    X = ones(10, 10)
    X_copy = copy(X)

    # Add noise without clipping
    add_gaussian_noise!(X; sigma=0.5, clip_at_zero=false)

    # X should have changed
    @test X != X_copy

    # With clipping, all entries must be >= 0
    X2 = fill(0.1, 10, 10)
    add_gaussian_noise!(X2; sigma=1.0, clip_at_zero=true)
    @test all(X2 .>= 0.0)

end


@testset "normalize_nonnegative!" begin
    
    # Without rescaling
    X = [-2.5 0.0 3.0; -1.0 5.0 1.0]
    normalize_nonnegative!(X; rescale=false)

    # Minimum should now be 0
    @test minimum(X) >= 0.0

    # With rescaling
    Y = [-2.5 0.0 3.0; -1.0 7.0 1.0]
    normalize_nonnegative!(Y; rescale=true)
    @test minimum(Y) >= 0.0
    @test maximum(Y) ≈ 1.0 

end

@testset "load_image_folder" begin
    
    mktempdir() do dir
        # Create two small grayscale images
        img1 = fill(Gray(0.2), 4, 4)
        img2 = fill(Gray(0.8), 4, 4)

        save(joinpath(dir, "img1.png"), img1)
        save(joinpath(dir, "img2.png"), img2)

        X, (h, w), filenames = load_image_folder(dir; pattern=".png", normalize=false)

        @test (h, w) == (4, 4)
        @test size(X) == (16, 2)  # 4×4 pixels, 2 images
        @test length(filenames) == 2
    end

end
