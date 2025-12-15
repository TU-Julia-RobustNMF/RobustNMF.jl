module StandardNMF

using LinearAlgebra, Random

export nmf

"""
    nmf(X; rank::Int=10, maxiter::Int=500, tol::Float64=1e-4, seed=nothing)

Standard NMF mit L2-Loss und multiplicative updates.

Gibt `(W, H, history)` zurück.
"""


function nmf(X; rank::Int = 10, maxiter::Int = 500)
        m, n = size(X)
        W = rand(m, rank)
        H = rand(rank, n)
        
        @assert minimum(X) ≥ 0 "X must be non-negative"


        # updating H and W 
        for iter in 1:maxiter
                H .*= (W' * X) ./ (W' * W * H .+ eps())
                W .*= (X * H') ./ (W * H * H' .+ eps())
        end
        return W, H, W*H
end


end # end module 