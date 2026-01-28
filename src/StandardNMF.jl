using LinearAlgebra, Random

"""
     nmf(X; rank::Int=10, maxiter::Int=500, tol::Float64=1e-4)

Compute a standard Non-negative Matrix Factorization (NMF) of the non-negative
data matrix `X ∈ ℝ^{m×n}` using multiplicative update rules for the squared Frobenius.

### Arguments
- `X`: non-negative data matrix of size `(m, n)`
- `rank`: factorization rank
- `maxiter`: maximum number of iterations
- `tol`: relative tolerance for stopping based on objective change
- `seed`: optional random seed for reproducibility

### Returns
- `W`: non-negative basis matrix of size `(m, rank)`
- `H`: non-negative coefficient matrix of size `(rank, n)`
- `history`: vector containing the squared Frobenius reconstruction error at each iteration

The reconstructed data matrix can be obtained as `W * H`.
"""
function nmf(X; rank::Int = 10, maxiter::Int = 500, tol::Float64 = 1e-4, seed=nothing)

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

