using LinearAlgebra, Random

"""
    nmf(X; rank::Int=10, maxiter::Int=500, tol::Float64=1e-4, seed::Int=0, eps_update::Float64=1e-12)

Compute a standard Non-negative Matrix Factorization (NMF) of the non-negative
data matrix `X in R^{m x n}` using multiplicative update rules and an L2 (Frobenius)
reconstruction loss.

### Returns
- `W`: non-negative basis matrix of size `(m, rank)`
- `H`: non-negative coefficient matrix of size `(rank, n)`
- `history`: vector containing the Frobenius reconstruction error at each iteration

The reconstructed data matrix can be obtained as `W * H`.
"""
function nmf(
    X;
    rank::Int = 10,
    maxiter::Int = 500,
    tol::Float64 = 1e-4,
    seed::Int = 0,
    eps_update::Float64 = 1e-12
)
    @assert minimum(X) >= 0 "X must be non-negative"

    Random.seed!(seed)

    m, n = size(X)
    W = rand(m, rank)
    H = rand(rank, n)

    history = zeros(Float64, maxiter)
    prev_err = Inf
    eps_conv = eps(Float64)

    # updating H and W
    for iter in 1:maxiter
        H .*= (W' * X) ./ (W' * W * H .+ eps_update)
        W .*= (X * H') ./ (W * H * H' .+ eps_update)

        # check how much it changed, if change too small --> stop
        err = norm(X - W * H)
        history[iter] = err
        if abs(prev_err - err) / (prev_err + eps_conv) < tol
            history = history[1:iter]
            break
        end
        prev_err = err
    end
    return W, H, history
end
