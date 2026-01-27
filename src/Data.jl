using Random, LinearAlgebra, Statistics
using FileIO: load
using ImageIO
using Images: channelview, colorview
using ColorTypes: Gray
using Base: basename


"""

    generate_synthetic_data(m::Int, n::Int; rank::Int=10, noise_level::Float64=0.0, seed=nothing)

Generate a non-negative matrix `X ∈ R^{m×n}` by sampling non-negative factors `W (m×rank)` and
`H (rank×n)` and returning `(X, W, H)`.

Optionally add Gaussian noise with standard deviation `noise_level` and clip the result at `0.0`
to keep `X` non-negative.

# Arguments
- `m::Int`: Number of rows of `X`.
- `n::Int`: Number of columns of `X`.

# Keyword Arguments
- `rank::Int=10`: Rank of the factorization.
- `noise_level::Float64=0.0`: Standard deviation of Gaussian noise.
- `seed`: Optional random seed for reproducibility.

# Returns
- `X::Matrix{Float64}`: Generated non-negative data matrix.
- `W::Matrix{Float64}`: Left factor.
- `H::Matrix{Float64}`: Right factor.

# Examples
```jldoctest
julia> X, W, H = generate_synthetic_data(20, 15; rank=3, seed=42);

julia> size(W), size(H)
((20, 3), (3, 15))

julia> minimum(X) >= 0
true
```
"""
function generate_synthetic_data(m::Int, n::Int; rank::Int=10, 
    noise_level::Float64=0.0, seed=nothing)
    
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    # Sample non-negative factors W and H from Uniform(0,1)
    W = rand(rng, m, rank)
    H = rand(rng, rank, n)

    # Construct non-negative data matrix
    X = W * H

    # Optionally add Gaussian noise and clip at 0.0
    if noise_level > 0
        noise = similar(X)          # same size and element type as X
        randn!(rng, noise)          # fill with Gaussian noise N(0,1)
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

# Arguments
- `X::AbstractMatrix`: Data matrix to be corrupted.

# Keyword Arguments
- `σ::Float64=0.1`: Noise standard deviation.
- `clip_at_zero::Bool=true`: Enforce non-negativity after corruption.

# Returns
- `X`: The modified input matrix.

# Examples
```jldoctest
julia> X = abs.(randn(5, 5));

julia> add_gaussian_noise!(X; σ=0.2);

julia> minimum(X) >= 0
true
```
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


# function add_gaussian_noise!(X::AbstractMatrix; σ::Float64=0.1, clip_at_zero::Bool=true, seed=nothing)
    
#     rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

#     # Allocate temporary noise buffer with same size/type as X
#     noise = similar(X)

#     # Fill noise with N(0, 1) samples and scale by σ
#     randn!(rng, noise)
#     noise .*= σ           

#     # Add noise to X in-place
#     X .+= noise        
    
#     # Optionally enforce non-negativity by clipping at 0.0
#     if clip_at_zero
#         @. X = max(X, 0.0)
#     end
    
#     return X
# end 

"""

    add_sparse_outliers!(X::AbstractMatrix; fraction::Float64=0.01, magnitude::Float64=5.0, 
    seed=nothing)

Add sparcse, large positive outliers to a fraction of the entries of `X` in-place.

`fraction` controls the proportion of entries that are modified.
Each selected entry is increased by a random value drawn from `Uniform(0, magnitude)`.
If `seed` is provided, the random choices are reproducible.

# Arguments
- `X::AbstractMatrix`: Data matrix to be corrupted.

# Keyword Arguments
- `fraction::Float64=0.01`: Fraction of entries to corrupt.
- `magnitude::Float64=5.0`: Maximum outlier amplitude.
- `seed`: Optional random seed.

# Returns
- `X`: The modified input matrix.

# Examples
```jldoctest
julia> X = zeros(10, 10);

julia> add_sparse_outliers!(X; fraction=0.05, seed=1);

julia> count(x -> x > 0, X) > 0
true
```
"""
function add_sparse_outliers!(X::AbstractMatrix; fraction::Float64=0.01, magnitude::Float64=5.0, 
    seed=nothing)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    # Determine how many entries to corrupt
    m, n = size(X)
    total = m * n
    k = max(1, round(Int, fraction * total))

    # Sample k random linear indices into X 4×4=16
    idx = rand(rng, 1:total, k)

    # Add large positive outliers at these positions
    X[idx] .+= magnitude .* rand(rng, k)
    
    return X
end


"""

    normalize_nonnegative!(X::AbstractMatrix; rescale::Bool=true)

Shift the matrix `X` in-place so that its minimum value becomes `0.0` if it is negative.
If `rescale` is `true`, also divide `X` by its maximum value so that all entries lie in the
interval `[0, 1]`.

# Arguments
- `X::AbstractMatrix`: Input matrix.

# Keyword Arguments
- `rescale::Bool=true`: Whether to divide by the maximum value.

# Returns
- `X`: The normalized matrix.

# Examples
```jldoctest
julia> X = [-1.0 2.0; 3.0 -4.0];

julia> normalize_nonnegative!(X);

julia> minimum(X), maximum(X)
(0.0, 1.0)
```
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

    load_image_folder(dir::AbstractString; pattern::AbstractString=".png", normalize::Bool=true)

Load all images in `dir` whose filenames match `pattern`, convert them to grayscale if needed,
flatten them, and stack them as columns of a data matrix `X`.

Returns a tuple `(X, (height, width), filenames)`, where:
- `X :: Matrix{Float64}` has one column per image,
- `(height, width)` is the original image size,
- `filenames` is a vector of the loaded base filenames.

If `normalize` is `true`, the matrix `X` is shifted and rescaled to be non-negative with entries
in `[0, 1]`.


# Arguments
- `dir::AbstractString`: Path to the image directory.

# Keyword Arguments
- `pattern::AbstractString="*.png"`: File extension filter.
- `normalize::Bool=true`: Normalize output matrix to `[0, 1]`.

# Returns
- `X::Matrix{Float64}`: One column per image.
- `(height, width)`: Original image dimensions.
- `filenames::Vector{String}`: Loaded file names.

# Examples
```jldoctest
julia> # X, size, names = load_image_folder("faces/")
```
"""
function load_image_folder(dir::AbstractString; pattern::AbstractString=".png", normalize::Bool=true)

    # Check if directory exists
    if !isdir(dir)
        error("Directory '$dir' does not exist or is not a directory")
    end

    # List all files in the directory (with full paths)
    files = sort(readdir(dir; join=true))

    # Keep only those whose path contains the pattern
    files = filter(f -> endswith(lowercase(f), lowercase(pattern)), files)

    if isempty(files)
        error("No files matching pattern '$pattern' found in $dir")
    end

    # Load and convert images to grayscale arrays
    imgs = Matrix{Float64}[]
    for f in files
        img = load(f)       # from FileIO/ImageIO

        # 2D grayscale Float64 matrix
        img_gray = Float64.(Array(Gray.(img)))  
        
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


