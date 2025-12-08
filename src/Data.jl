module DataType

using Random, LinearAlgebra, Statistics

"""
Generate a non-negative matrix X ∈ R^{m×n} by sampling non-negative factors W (m×r)
and H (r×n) and returning (X, W, H)

Optionally add Gaussian noise (clipped at 0 to keep non-negativity).
"""
function generate_synthetic_data(m::Int, n::Int; rank::Int=10, 
    noise_level::Float64=0.0, seed=nothing)
    
    # Only call if seed is given
    if seed !== nothing
        Random.seed!(seed)
    end

    W = rand(m, rank)
    H = rand(rank, n)
    X = W * H

    if noise_level > 0
        noise = similar(X)          # Create matrix with uninitialized values
        randn!(noise)               # Fill with Gaussian noise N(0,1)
        X .+= noise_level .* noise  # Add scaled noise
        @. X = max(X, 0.0)          # Clip to keep non-negative
    end

    return X, W, H
end
end


"""
In-place add Gaussian noise with std σ to X.
Optionally clip at 0 to maintain non-negativity.
"""
function add_gaussian_noise!(X::AbstractMatrix; σ::Float64=0.1, clip_at_zero::Bool=true)
    
    noise = similar(X)
    randn!(noise)
    noise .*= σ           
    X .+= noise        
    
    if clip_at_zero
        @. X = max(X, 0.0)
    end
    
    return X
end

"""
Add sparse, large positive outliers to a fraction of entries of X.
"""
function add_sparse_outliers!(X::AbstractMatrix; fraction::Float64-0.01, magnitude::Float64, 
    seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    m, n = size(X)
    total = m * n
    k = max(1, round(Int, fraction * total))

    # random linear indices
    idx = rand(1:total, k)
    X[idx] .+= magniute .* rand(x)
    
    return X
end

# Example:
# --------
# X, W,_true, H_true = generate_synthetic_data(100, 80; rank=8)
# add_gaussian_noise!(X; σ=0.2)
# add_sparse_outliers!(X; fraction=0.02, magnitude=10.0)

