using LinearAlgebra: norm, mul!
using Random: Random, MersenneTwister, rand

"""
    nmf(X; rank::Int=10, maxiter::Int=500, tol::Real=1e-4, seed=nothing)

Compute a standard non-negative matrix factorization (NMF) of `X` using
multiplicative update rules with squared Frobenius reconstruction loss.

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix of size `(m, n)`.

# Keyword Arguments
- `rank::Int=10`: Target factorization rank.
- `maxiter::Int=500`: Maximum number of iterations.
- `tol::Real=1e-4`: Relative tolerance for stopping based on objective change.
- `seed=nothing`: Optional random seed for reproducible initialization.

# Returns
- `W::Matrix{Float64}`: Non-negative basis matrix of size `(m, rank)`.
- `H::Matrix{Float64}`: Non-negative coefficient matrix of size `(rank, n)`.
- `history::Vector{Float64}`: Squared Frobenius reconstruction error per iteration.

# Side Effects
- None. (The function does not modify `X`.)

# Errors
- `ArgumentError`: If `X` contains negative entries or parameters are invalid.

# Notes
- Uses multiplicative updates with a small epsilon for numerical stability.
- Convergence check is based on relative change in the objective.
- The reconstruction can be obtained as `W * H`.
- Input data should be non-negative.

# Examples
```jldoctest
julia> X, _, _ = generate_synthetic_data(20, 15; rank=3, seed=42);

julia> W, H, history = nmf(X; rank=3, maxiter=200, tol=1e-5, seed=42);

julia> size(W), size(H), length(history) > 0
((20, 3), (3, 15), true)
```
"""
function nmf(X; rank::Int = 10, maxiter::Int = 500, tol::Real = 1e-4, seed=nothing)
    if any(x -> x < 0, X)
        throw(ArgumentError("X must be non-negative for NMF (found negative entries)."))
    end
    if rank <= 0
        throw(ArgumentError("rank must be positive (got rank=$rank)."))
    end
    if maxiter <= 0
        throw(ArgumentError("maxiter must be positive (got maxiter=$maxiter)."))
    end
    if tol <= 0
        throw(ArgumentError("tol must be positive (got tol=$tol)."))
    end

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    m, n = size(X)
    T = eltype(X)
    W = rand(rng, T, m, rank)
    H = rand(rng, T, rank, n)

    ϵ = eps(T)

    history = T[]
    prev_obj = T(Inf)

    WtW = similar(W, rank, rank)
    HtH = similar(H, rank, rank)
    WtX = similar(H, rank, n)
    XHt = similar(W, m, rank)
    WtWH = similar(H, rank, n)
    WHHt = similar(W, m, rank)
    WH = similar(X, m, n)
    R = similar(X, m, n)

    # Updating H and W
    for iter in 1:maxiter
        # H update: (W' * X) ./ (W' * W * H + ϵ)
        mul!(WtX, W', X)
        mul!(WtW, W', W)
        mul!(WtWH, WtW, H)
        @. H *= WtX / (WtWH + ϵ)

        # W update: (X * H') ./ (W * H * H' + ϵ)
        mul!(XHt, X, H')
        mul!(HtH, H, H')
        mul!(WHHt, W, HtH)
        @. W *= XHt / (WHHt + ϵ)

        # Check how much it changed, if change too small --> stop
        mul!(WH, W, H)
        @. R = X - WH
        obj = norm(R)^2  # in julia norm of matrix is frobenius norm by default
        push!(history, obj)

        # Relative change stopping criterion
        if abs(prev_obj - obj) / (prev_obj + ϵ) < tol
            break
        end
        prev_obj = obj
    end
    return W, H, history
end
