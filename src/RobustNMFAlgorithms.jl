module RobustNMFAlgorithms

using LinearAlgebra

export robust_nmf

"""
    robust_nmf(X; rank=10, maxiter=500, tol=1e-4, epsval=1e-9, weight_floor=1e-6)

Robuste NMF via IRLS (L1-ähnlich): iterativ gewichtete Least-Squares Updates.

Minimiert näherungsweise ‖X - W*H‖₁, indem Gewichte wᵢⱼ = 1 / (|Rᵢⱼ| + epsval) genutzt werden,
wobei R = X - W*H.

Returns: W, H, history
"""
function robust_nmf(X;
    rank::Int = 10,
    maxiter::Int = 500,
    tol::Float64 = 1e-4,
    epsval::Float64 = 1e-9,
    weight_floor::Float64 = 1e-6
)
    @assert minimum(X) ≥ 0 "X must be non-negative"

    m, n = size(X)
    W = rand(m, rank)
    H = rand(rank, n)

    history = zeros(Float64, maxiter)
    prev_obj = Inf

    for iter in 1:maxiter
        # Current reconstruction and residuals
        WH = W * H
        R  = X .- WH

        # IRLS weights (downweight large residuals/outliers)
        w = 1.0 ./ (abs.(R) .+ epsval)
        w .= max.(w, weight_floor)  # avoid extremely tiny weights

        # Multiplicative updates for weighted Frobenius: min Σ wᵢⱼ (Xᵢⱼ - (WH)ᵢⱼ)²
        WX = w .* X
        WWH = w .* WH

        H .*= (W' * WX) ./ (W' * WWH .+ epsval)
        WH = W * H
        WWH = w .* WH
        W .*= (WX * H') ./ (WWH * H' .+ epsval)

        # Robust objective proxy: L1 loss (true target) for monitoring
        WH = W * H
        obj = sum(abs.(X .- WH))
        history[iter] = obj

        # Stopping
        if abs(prev_obj - obj) / (prev_obj + epsval) < tol
            return W, H, history[1:iter]
        end
        prev_obj = obj
    end

    return W, H, history
end

end # module
