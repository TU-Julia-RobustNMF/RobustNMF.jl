using LinearAlgebra: norm
using Random: Random, MersenneTwister, rand

"""
    nmf(X; rank::Int=10, maxiter::Int=500, tol::Float64=1e-4, seed=nothing)

Compute a standard non-negative matrix factorization (NMF) of `X` using
multiplicative update rules with squared Frobenius reconstruction loss.

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix of size `(m, n)`.

# Keyword Arguments
- `rank::Int=10`: Target factorization rank.
- `maxiter::Int=500`: Maximum number of iterations.
- `tol::Float64=1e-4`: Relative tolerance for stopping based on objective change.
- `seed=nothing`: Optional random seed for reproducible initialization.

# Returns
- `W::Matrix{Float64}`: Non-negative basis matrix of size `(m, rank)`.
- `H::Matrix{Float64}`: Non-negative coefficient matrix of size `(rank, n)`.
- `history::Vector{Float64}`: Squared Frobenius reconstruction error per iteration.

# Side Effects
- None. (The function does not modify `X`.)

# Errors
- None.

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

        rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

        m, n = size(X)
        W = rand(rng, m, rank)
        H = rand(rng, rank, n)

        # @assert minimum(X) >= 0 "X must be non-negative"

        ϵ = eps(Float64)

        history = Float64[]
        prev_obj = Inf
        

        # updating H and W 
        for iter in 1:maxiter
                H .*= (W' * X) ./ (W' * W * H .+ ϵ)
                W .*= (X * H') ./ (W * H * H' .+ ϵ)

                # check how much it changed, if change too small --> stop
                obj = norm(X - W * H)^2 # in julia norm of matrix is frobenius norm by default
                push!(history, obj)

                # Relative change stopping criterion
                if abs(prev_obj - obj) / (prev_obj + ϵ) < tol
                        break
                end
                prev_obj = obj
        end
        return W, H, history
end
