using LinearAlgebra
using Random
using Statistics

"""
    robust_nmf(X; rank=2, maxiter=50, tol=1e-4, eps_weight=1e-6, eps_update=1e-12, seed=0)

Very simple robust NMF via IRLS style weighting plus multiplicative updates.
Stops early when the relative change in mean absolute error falls below `tol`.

Returns:
    W, H, history
where history[k] = Frobenius reconstruction error at iteration k.
"""
function robust_nmf(
    X::AbstractMatrix{<:Real};
    rank::Int = 2,
    maxiter::Int = 50,
    tol::Float64 = 1e-4,
    eps_weight::Float64 = 1e-6,
    eps_update::Float64 = 1e-12,
    seed::Int = 0
)
    @assert minimum(X) >= 0 "X must be non negative"

    Random.seed!(seed)

    m, n = size(X)
    W = rand(m, rank)
    H = rand(rank, n)

    history = zeros(Float64, maxiter)
    prev_err = Inf
    eps_conv = eps(Float64)

    for iter in 1:maxiter
        # 1) Current reconstruction
        WH = W * H

        # 2) Residuals (Reconstruction difference to X)
        R = X .- WH

        # 3) IRLS weights, downweight large residuals
        V = 1.0 ./ (abs.(R) .+ eps_weight)

        # 4) Weighted multiplicative update for H
        numerH = W' * (V .* X)
        denomH = W' * (V .* (W * H)) .+ eps_update
        H .*= numerH ./ denomH

        # 5) Weighted multiplicative update for W
        numerW = (V .* X) * H'
        denomW = (V .* (W * H)) * H' .+ eps_update
        W .*= numerW ./ denomW

        # 6) Monitor reconstruction error (Frobenius norm) for comparability with nmf
        err = norm(X .- (W * H))
        history[iter] = err
        if abs(prev_err - err) / (prev_err + eps_conv) < tol
            history = history[1:iter]
            break
        end
        prev_err = err
    end

    return W, H, history
end

