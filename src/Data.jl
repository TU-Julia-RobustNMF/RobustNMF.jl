module Data

using Random, LinearAlgebra, Statistics
using FileIO: load
using ImageIO
using Images: channelview, colorview
using ColorTypes: Gray
using Base: basename

export
generate_synthetic_data, 
add_gaussian_noise!, 
add_sparse_outliers!, 
normalize_nonnegative!, 
load_image_folder


"""

    generate_synthetic_data(m::Int, n::Int; rank::Int=10, noise_level::Float64=0.0, seed=nothing)

Generate a non-negative matrix `X ∈ R^{m×n}` by sampling non-negative factors `W (m×rank)` and
`H (rank×n)` and returning `(X, W, H)`.

Optionally add Gaussian noise with standard deviation `noise_level` and clip the result at `0.0`
to keep `X` non-negative.
"""
function generate_synthetic_data(m::Int, n::Int; rank::Int=10, 
    noise_level::Float64=0.0, seed=nothing)
    
    # Set RNG seed only if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Sample non-negative factors W and H from Uniform(0,1)
    W = rand(m, rank)
    H = rand(rank, n)

    # Construct non-negative data matrix
    X = W * H

    # Optionally add Gaussian noise and clip at 0.0
    if noise_level > 0
        noise = similar(X)          # same size and element type as X
        randn!(noise)               # fill with Gaussian noise N(0,1)
        X .+= noise_level .* noise  # add scaled noise
        @. X = max(X, 0.0)          # clip negatives to 0.0
    end

    return X, W, H
end


"""

    add_gaussian_noise!(X::AbstractMatrix; σ::Float64=0.1, clip_at_zero::Bool=true)

Add Gaussian noise with standard deviation `σ` to the matrix `X` in-place.

If `clip_at_zero` is `true`, replace all negative entries of `X` with `0.0` after adding noise,
to preserve non-negativity.
"""
function add_gaussian_noise!(X::AbstractMatrix; σ::Float64=0.1, clip_at_zero::Bool=true)
    
    # Allocate temporary noise buffer with same size/type as X
    noise = similar(X)

    # Fill noise with N(0, 1) samples and scale by σ
    randn!(noise)
    noise .*= σ           

    # Add noise to X in-place
    X .+= noise        
    
    # Optionally enforce non-negativity by clipping at 0.0
    if clip_at_zero
        @. X = max(X, 0.0)
    end
    
    return X
end


"""

    add_sparse_outliers!(X::AbstractMatrix; fraction::Float64=0.01, magnitude::Float64=5.0, 
    seed=nothing)

Add sparcse, large positive outliers to a fraction of the entries of `X` in-place.

`fraction` controls the proportion of entries that are modified.
Each selected entry is increased by a random value drawn from `Uniform(0, magnitude)`.
If `seed` is provided, the random choices are reproducible.
"""
function add_sparse_outliers!(X::AbstractMatrix; fraction::Float64=0.01, magnitude::Float64=5.0, 
    seed=nothing)

    # Set RNG seed only if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # Determine how many entries to corrupt
    m, n = size(X)
    total = m * n
    k = max(1, round(Int, fraction * total))

    # Sample k random linear indices into X 4×4=16
    idx = rand(1:total, k)

    # Add large positive outliers at these positions
    X[idx] .+= magnitude .* rand(k)
    
    return X
end


"""

    normalize_nonnegative!(X::AbstractMatrix; rescale::Bool=true)

Shift the matrix `X` in-place so that its minimum value becomes `0.0` if it is negative.
If `rescale` is `true`, also divide `X` by its maximum value so that all entries lie in the
interval `[0, 1]`.
"""
function normalize_nonnegative!(X::AbstractMatrix; rescale::Bool=true)

    # Shift X so that its minimum value becomes 0.0 (if needed)
    min_val = minimum(X)
    if min_val < 0
        X .-= min_val
    end

    # Optionally rescale X so that maximum becomes 1.0
    if rescale
        max_val = maximum(X)
        if max_val > 0
            X ./= max_val
        end
    end

    return X

end


"""

    load_image_folder(dir::AbstractString; pattern::AbstractString="*.png", normalize::Bool=true)

Load all images in `dir` whose filenames match `pattern`, convert them to grayscale if needed,
flatten them, and stack them as columns of a data matrix `X`.

Returns a tuple `(X, (height, width), filenames)`, where:
- `X :: Matrix{Float64}` has one column per image,
- `(height, width)` is the original image size,
- `filenames` is a vector of the loaded base filenames.

If `normalize` is `true`, the matrix `X` is shifted and rescaled to be non-negative with entries
in `[0, 1]`.
"""
function load_image_folder(dir::AbstractString; pattern::AbstractString="*.png", normalize::Bool=true)

    # add condition to check if directory exists
    # ------------------------------------------

    # List all files in the directory (with full paths)
    files = sort(readdir(dir; join=true))

    # Keep only those whose path contains the pattern
    files = filter(f -> endswith(lowercase(f), lowercase(pattern)), files)

    if isempty(files)
        error("No files matching pattern '$pattern' found in $dir")
    end

    # Load and convert images to grayscale arrays
    imgs = Any[]
    for f in files
        img = load(f)       # from FileIO/ImageIO

        # Convert to grayscale and Float64
        # colorview(Gray, img) ensures grayscale, channelview gives a 2D array
        img_gray = float.(channelview(colorview(Gray, img)))

        # img_gray is 2D (height, width)
        push!(imgs, img_gray)
    end

    # Check if all images have the same size
    h, w = size(imgs[1])
    for img in imgs
        size(img) == (h, w) || error("All images must have same size")
    end

    # Flatten and stack as columns in X
    num = length(imgs)
    X = zeros(h * w, num)
    for (j, img) in enumerate(imgs)
        X[:, j] .= vec(img)
    end

    # Optionally normalize to [0, 1] and non-negative
    if normalize
        normalize_nonnegative!(X)
    end

    # Return base.filenames (without directory)
    filenames = basename.(files)
    
    return X, (h, w), filenames

end


end  # module Data