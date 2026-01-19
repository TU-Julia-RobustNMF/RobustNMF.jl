using LinearAlgebra
using Random
using Statistics

"""
    robust_nmf(X; rank=2, maxiter=50, tol=1e-4, eps_weight=1e-6, eps_update=1e-12, seed=0)

Very simple robust NMF via IRLS style weighting plus multiplicative updates.
Stops early when the relative change in mean absolute error falls below `tol`.

Returns:
    W, H, history
where history[k] = mean(abs.(X - W*H)) at iteration k.
"""
function robust_nmf(
    X::AbstractMatrix{<:Real};
    rank::Int = 2,
    maxiter::Int = 50,
    tol::Float64 = 1e-4,
    eps_weight::Float64 = 1e-2,  # Increased from 1e-6 for stability
    eps_update::Float64 = 1e-10, # Increased from 1e-12
    seed::Int = 0
)
    @assert minimum(X) >= 0 "X must be non-negative"
    @assert rank > 0 "rank must be positive"
    @assert maxiter > 0 "maxiter must be positive"

    Random.seed!(seed)

    m, n = size(X)
    
    # Initialize with small positive random values
    W = rand(m, rank) .* 0.5 .+ 0.1
    H = rand(rank, n) .* 0.5 .+ 0.1
    
    # Normalize initial factors
    for i in 1:rank
        W[:, i] ./= (norm(W[:, i]) + eps_update)
    end

    history = zeros(Float64, maxiter)
    prev_err = Inf
    eps_conv = eps(Float64)

    for iter in 1:maxiter
        # 1) Current reconstruction
        WH = W * H

        # 2) Compute Frobenius error for history
        err = norm(X - WH)
        history[iter] = err

        # 3) Compute residuals
        R = X .- WH

        # 4) IRLS weights - downweight large residuals
        # Use sqrt of absolute residuals for softer weighting
        abs_R = abs.(R)
        V = 1.0 ./ (sqrt.(abs_R) .+ eps_weight)
        
        # Normalize weights to prevent numerical issues
        V ./= (mean(V) + eps_conv)

        # 5) Weighted multiplicative update for H
        numerH = W' * (V .* X)
        denomH = W' * (V .* WH) .+ eps_update
        H .*= numerH ./ denomH
        
        # Ensure non-negativity and prevent zeros
        @. H = max(H, eps_update)

        # 6) Weighted multiplicative update for W
        WH = W * H  # Recompute after H update
        numerW = (V .* X) * H'
        denomW = (V .* WH) * H' .+ eps_update
        W .*= numerW ./ denomW
        
        # Ensure non-negativity and prevent zeros
        @. W = max(W, eps_update)

        # 7) Check convergence
        if abs(prev_err - err) / (prev_err + eps_conv) < tol
            history = history[1:iter]
            break
        end
        prev_err = err
    end

    return W, H, history
end
