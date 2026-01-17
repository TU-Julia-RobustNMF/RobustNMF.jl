module StandardNMF

using LinearAlgebra, Random

export nmf

"""
     nmf(X; rank::Int=10, maxiter::Int=500, tol::Float64=1e-4)

Compute a standard Non-negative Matrix Factorization (NMF) of the non-negative
data matrix `X ∈ ℝ^{m×n}` using multiplicative update rules and an L2 (Frobenius)
reconstruction loss. 

### Returns
- `W`: non-negative basis matrix of size `(m, rank)`
- `H`: non-negative coefficient matrix of size `(rank, n)`
- `history`: vector containing the Frobenius reconstruction error at each iteration

The reconstructed data matrix can be obtained as `W * H`.
"""
function nmf(X; rank::Int = 10, maxiter::Int = 500, tol::Float64 = 1e-4)
        m, n = size(X)
        W = rand(m, rank)
        H = rand(rank, n)
        @assert minimum(X) >= 0 "X must be non-negative"

        history = zeros(maxiter)
        prev_err = Inf
        ϵ = eps(Float64)

        # updating H and W 
        for iter in 1:maxiter
                H .*= (W' * X) ./ (W' * W * H .+ ϵ)
                W .*= (X * H') ./ (W * H * H' .+ ϵ)

                # check how much it changed, if change too small --> stop
                err = norm(X - W * H) # in julia norm of matrix is frobenius norm by default
                history[iter] = err
                if abs(prev_err - err) / (prev_err + ϵ) < tol
                        history = history[1:iter]
                        break
                end
                prev_err = err
        end
        return W, H, history
end

end # end module 
