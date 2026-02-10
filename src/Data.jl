using Random: Random, MersenneTwister, rand, randn!
using FileIO: load
using ImageIO
using ColorTypes: Gray
using Base: basename


"""
    generate_synthetic_data(m::Int, n::Int; rank::Int=10, noise_level::Real=0.0, seed=nothing)

Generate a synthetic non-negative data matrix `X` as `W * H` with random non-negative factors.
Optionally adds Gaussian noise and clips negative values to keep `X ≥ 0`.

# Arguments
- `m::Int`: Number of rows of `X`.
- `n::Int`: Number of columns of `X`.

# Keyword Arguments
- `rank::Int=10`: Rank of the factorization.
- `noise_level::Real=0.0`: Standard deviation of Gaussian noise.
- `seed=nothing`: Optional random seed for reproducibility.

# Returns
- `X::Matrix{Float64}`: Generated non-negative data matrix.
- `W::Matrix{Float64}`: Left factor of size `(m, rank)`.
- `H::Matrix{Float64}`: Right factor of size `(rank, n)`.

# Side Effects
- None.

# Errors
- None.

# Notes
- Uses a local RNG seeded with `seed` (if provided) for deterministic output.
- When `noise_level > 0`, the result is clipped at `0.0` to enforce non-negativity.

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
    noise_level::Real=0.0, seed=nothing)
    
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
        @. X = max(X, zero(eltype(X)))  # clip negatives to zero
    end

    return X, W, H
end


"""
    add_gaussian_noise!(X::AbstractMatrix; σ::Real=0.1, clip_at_zero::Bool=true)

Add Gaussian noise with standard deviation `σ` to the matrix `X` in-place.
Optionally clip negative entries to preserve non-negativity.

# Arguments
- `X::AbstractMatrix`: Data matrix to be corrupted.

# Keyword Arguments
- `σ::Real=0.1`: Noise standard deviation.
- `clip_at_zero::Bool=true`: Enforce non-negativity after corruption.

# Returns
- `X`: The modified input matrix (in-place).

# Side Effects
- Modifies `X` in-place.

# Errors
- None.

# Notes
- Uses `randn!` to generate Gaussian noise.
- If `clip_at_zero=true`, negative values are replaced by `0.0`.

# Examples
```jldoctest
julia> X = abs.(randn(5, 5));

julia> add_gaussian_noise!(X; σ=0.2);

julia> minimum(X) >= 0
true
```
"""
function add_gaussian_noise!(X::AbstractMatrix; σ::Real=0.1, clip_at_zero::Bool=true)
    
    # Allocate temporary noise buffer with same size/type as X
    noise = similar(X)

    # Fill noise with N(0, 1) samples and scale by σ
    randn!(noise)
    noise .*= σ           

    # Add noise to X in-place
    X .+= noise        
    
    # Optionally enforce non-negativity by clipping at zero
    if clip_at_zero
        @. X = max(X, zero(eltype(X)))
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
    add_sparse_outliers!(X::AbstractMatrix; fraction::Real=0.01, magnitude::Real=5.0, seed=nothing)

Add sparse, large positive outliers to a fraction of the entries of `X` in-place.

# Arguments
- `X::AbstractMatrix`: Data matrix to be corrupted.

# Keyword Arguments
- `fraction::Real=0.01`: Fraction of entries to corrupt.
- `magnitude::Real=5.0`: Maximum outlier amplitude.
- `seed=nothing`: Optional random seed.

# Returns
- `X`: The modified input matrix (in-place).

# Side Effects
- Modifies `X` in-place.

# Errors
- None.

# Notes
- Corrupts `max(1, round(Int, fraction * m*n))` entries.
- Added outliers are sampled from `Uniform(0, magnitude)`.

# Examples
```jldoctest
julia> X = zeros(10, 10);

julia> add_sparse_outliers!(X; fraction=0.05, seed=1);

julia> count(>(0.0), X) > 0
true
```
"""
function add_sparse_outliers!(X::AbstractMatrix; fraction::Real=0.01, magnitude::Real=5.0,
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

Shift `X` in-place so that the minimum becomes `0.0`, and optionally rescale to `[0, 1]`.

# Arguments
- `X::AbstractMatrix`: Input matrix.

# Keyword Arguments
- `rescale::Bool=true`: Whether to divide by the maximum value after shifting.

# Returns
- `X`: The normalized matrix (in-place).

# Side Effects
- Modifies `X` in-place.

# Errors
- None.

# Notes
- If `rescale=true` and `maximum(X) == 0`, rescaling is skipped.

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

Load images from a folder, convert to grayscale, flatten, and stack them as columns of `X`.

# Arguments
- `dir::AbstractString`: Path to the image directory.

# Keyword Arguments
- `pattern::AbstractString=".png"`: File extension filter (matched via `endswith`).
- `normalize::Bool=true`: Normalize output matrix to `[0, 1]`.

# Returns
- `X::Matrix{Float64}`: One column per image.
- `(height, width)`: Original image dimensions.
- `filenames::Vector{String}`: Loaded base file names.

# Side Effects
- Reads image files from disk.

# Errors
- `ErrorException`: If the directory does not exist or no files match `pattern`.
- `ErrorException`: If images have inconsistent sizes.

# Notes
- Images are converted to grayscale and stored as `Float64`.
- If `normalize=true`, `normalize_nonnegative!` is applied to `X`.

# Examples
```jldoctest
julia> # X, size, names = load_image_folder(\"faces/\")
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

    # Load first image to determine size and preallocate X
    first_img = load(files[1])
    first_gray = Float64.(Array(Gray.(first_img)))
    h, w = size(first_gray)

    num = length(files)
    X = zeros(h * w, num)
    X[:, 1] .= vec(first_gray)

    # Load remaining images, validate size, and fill X
    for (j, f) in enumerate(files[2:end], start=2)
        img = load(f)
        img_gray = Float64.(Array(Gray.(img)))
        size(img_gray) == (h, w) || error("All images must have same size")
        X[:, j] .= vec(img_gray)
    end

    # Optionally normalize to [0, 1] and non-negative
    if normalize
        normalize_nonnegative!(X)
    end

    # Return base.filenames (without directory)
    filenames = basename.(files)
    
    return X, (h, w), filenames

end
