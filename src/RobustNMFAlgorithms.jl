using LinearAlgebra
using Random


"""
    l21norm(X)

Compute the L2,1-norm of matrix X.
The L2,1-norm is the sum of the L2-norms of each column.

# Arguments
- `X::AbstractMatrix`: Input matrix

# Returns
- Scalar value: sum of L2-norms of columns

# Examples
```julia
X = [1.0 2.0; 3.0 4.0]
l21norm(X)  # Returns norm([1,3]) + norm([2,4])
```
"""
function l21norm(X::AbstractMatrix)
    return sum(norm(X[:, i]) for i in 1:size(X, 2))
end


"""
    update(X, F, G; eps_update=1e-10)

Perform one iteration of L2,1-NMF multiplicative updates.

The L2,1-norm promotes row sparsity in the residual matrix, making the
algorithm robust to sample-wise (column-wise) outliers.

# Arguments
- `X::AbstractMatrix`: Data matrix (m × n)
- `F::AbstractMatrix`: Current basis matrix (m × rank)
- `G::AbstractMatrix`: Current coefficient matrix (rank × n)

# Keyword Arguments
- `eps_update::Float64=1e-10`: Small constant for numerical stability

# Returns
- `F_new::Matrix{Float64}`: Updated basis matrix
- `G_new::Matrix{Float64}`: Updated coefficient matrix
"""
function update(X::AbstractMatrix, F::AbstractMatrix, G::AbstractMatrix; 
                    eps_update::Float64=1e-10)
    
    m, n = size(X)
    rank = size(F, 2)
    
    # Compute diagonal weight matrix D
    # d[i] = 1 / (2 * ||x_i - F*g_i||_2)
    d = zeros(n)
    for i in 1:n
        residual_norm = norm(X[:, i] - F * G[:, i])
        d[i] = 1.0 / (2.0 * residual_norm + eps_update)
    end
    D = Diagonal(d)
    
    # Update F using multiplicative update rule
    F_new = similar(F)
    F1 = X * D * G'
    F2 = F * G * D * G' .+ eps_update
    
    for j in 1:m
        for k in 1:rank
            F_new[j, k] = F[j, k] * (F1[j, k] / F2[j, k])
        end
    end
    # QUESTION: why not F_new = F .* (F1 ./ F2)


    # Ensure non-negativity
    @. F_new = max(F_new, eps_update)
    
    # Update G using multiplicative update rule
    G_new = similar(G)
    G1 = F_new' * X * D
    G2 = F_new' * F_new * G * D .+ eps_update
    
    for k in 1:rank
        for i in 1:n
            G_new[k, i] = G[k, i] * (G1[k, i] / G2[k, i])
        end
    end
# QUESTION: why not G_new = G .* (G1 ./ G2)

    
    # Ensure non-negativity
    @. G_new = max(G_new, eps_update)
    
    return F_new, G_new
end


"""
    robustnmf_l21(X; rank=10, maxiter=500, tol=1e-4, seed=nothing)

L2,1-Norm Regularized Non-negative Matrix Factorization.

Minimizes: ||X - FG||_{2,1} where the L2,1-norm promotes robustness
to sample-wise outliers (entire corrupted columns in `X`).

NOTE: The course PDF for this project specifies robust NMF via **L1**, **Huber**, or **Itakura-Saito**, 
we keep this implementation temporarily to avoid breaking existing code during the migration.
"""
function robustnmf_l21(X::AbstractMatrix{<:Real}; 
                 rank::Int=10, 
                 maxiter::Int=500, 
                 tol::Float64=1e-4,
                 seed::Union{Int,Nothing}=nothing)
    
    @assert minimum(X) >= 0 "X must be non-negative"
    @assert rank > 0 "rank must be positive"
    @assert maxiter > 0 "maxiter must be positive"
    
    # Set random seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end
    
    m, n = size(X)
    
    # Initialize F and G with random non-negative values
    F = rand(m, rank) .* 0.5 .+ 0.1
    G = rand(rank, n) .* 0.5 .+ 0.1
    
    # Track convergence history
    history = zeros(Float64, maxiter)
    
    # Iterative updates
    for iter in 1:maxiter
        # Perform one L2,1-NMF update
        F, G = update(X, F, G)
        
        # Compute L2,1-norm error
        error = l21norm(X - F * G)
        history[iter] = error
        
        # Check convergence
        if error < tol # QUESTION: why not relative change: abs(prev_err - err) / (prev_err + ϵ) < tol
            history = history[1:iter]
            break
        end
    end
    return F, G, history
end


"""
    robustnmf_huber(X; rank=10, maxiter=500, tol=1e-4, delta=1.0, seed=nothing)

Implements a robust NMF variant using the Huber loss on the reconstruction residuals:
    R = X - W*H

Huber loss (element-wise) with threshold δ:
- if |r| ≤ δ: 0.5 * r^2
- else:       δ * (|r| - 0.5*δ)

We optimize it using an IRLS-style weighted least squares approach:
1) compute residual R
2) compute weights Ω = huber_weights(R, δ)
3) perform weighted multiplicative updates (update_huber)
4) track huber_loss(R, δ) in `history`
5) stop when relative change in objective is below `tol`

### Arguments
- `X`: non-negative data matrix of size `(m, n)`
- `rank`: factorization rank`
- `maxiter`: maximum number of iterations
- `tol`: relative tolerance for stopping based on objective change
- `delta`: Huber threshold δ (must be > 0)
- `seed`: optional random seed for reproducibility

### Returns
- `W`: non-negative basis matrix of size `(m, rank)`
- `H`: non-negative coefficient matrix of size `(rank, n)`
- `history`: vector of Huber objective values per iteration 
"""
function robustnmf_huber(
    X::AbstractMatrix{<:Real};
    rank::Int = 10,
    maxiter::Int = 500,
    tol::Float64 = 1e-4,
    delta::Float64 = 1.0,
    seed::Union{Int,Nothing} = nothing)

    # --- Input validation (robust NMF requires non-negative data) ---
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
    if delta <= 0
        throw(ArgumentError("delta must be > 0 for Huber loss (got delta=$delta)."))
    end

    # --- Reproducible initialization using a local RNG ---
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    m, n = size(X)

    # Initialize W and H with random non-negative values
    W = rand(rng, m, rank)
    H = rand(rng, rank, n)

    # Small constant to avoid division by zero in multiplicative updates
    ϵ = eps(Float64)

    # Track objective history (Huber loss values)
    history = Float64[]
    prev_obj = Inf

    # --- Main optimization loop ---
    for iter in 1:maxiter
        # Compute residual with current factors
        R = X - W * H

        # Compute current objective (Huber loss)
        obj = huber_loss(R, delta)
        push!(history, obj)

        # Stopping criterion: relative change in objective
        # Note: first iteration won't stop because prev_obj = Inf
        if abs(prev_obj - obj) / (prev_obj + ϵ) < tol
            break
        end
        prev_obj = obj

        # Compute IRLS weights from current residual
        Ω = huber_weights(R, delta; ϵ=ϵ)

        # Perform one weighted multiplicative update step
        W, H = update_huber(X, W, H, Ω; ϵ=ϵ)

        # Finite check to catch numerical issues early
        if !(all(isfinite, W) && all(isfinite, H))
            throw(ErrorException("Numerical instability encountered: W/H contain NaN or Inf."))
        end
    end

    return W, H, history
end

"""
    huber_loss(R, delta; ϵ=eps(Float64))

Compute the **Huber loss** for a residual matrix `R`.

Huber loss is robust to outliers:
- For small residuals it behaves like squared error (L2).
- For large residuals it behaves like absolute error (L1), reducing the impact of outliers.

Element-wise definition for residual r:
- if |r| ≤ δ: 0.5 * r^2
- else:       δ * (|r| - 0.5*δ)

Returns the **sum** over all entries of R.
"""
function huber_loss(R::AbstractMatrix{<:Real}, delta::Real; ϵ::Real = eps(Float64))::Float64
    # Basic parameter validation:
    # delta controls where we transition from quadratic (L2) to linear (L1-like).
    if delta <= 0
        throw(ArgumentError("delta must be > 0 for Huber loss (got delta=$delta)."))
    end

    # Convert delta once to Float64 to avoid repeated conversions inside loops.
    δ = Float64(delta)

    # Accumulate total loss in Float64 for numerical stability
    total = 0.0

    # Loop explicitly for performance and to avoid temporary allocations
    @inbounds for r in R
        # Residual magnitude
        ar = abs(Float64(r))

        if ar <= δ
            # Quadratic region: 0.5 * r^2
            total +=0.5 * ar * ar
        else
            # Linear region: δ*(|r| - 0.5*δ)
            total += δ * (ar - 0.5 * δ)
        end
    end

    return total
end


"""
    huber_weights(R, delta; ϵ=eps(Float64))

Compute the **Huber IRLS weights** matrix Ω for a residual matrix `R`.

We use an iteratively reweighted least squares (IRLS) interpretation:
- Small residuals get weight 1.0 (quadratic region).
- Large residuals get weight δ / (|r| + ϵ), which downweights outliers.

Element-wise:
- if |r| ≤ δ: w = 1
- else:       w = δ / (|r| + ϵ)

Returns Ω with the same size as R.
"""
function huber_weights(R::AbstractMatrix{<:Real}, delta::Real; ϵ::Real = eps(Float64))::Matrix{Float64}
    # Validate delta: must be positive to define a Huber threshold.
    if delta <= 0
        throw(ArgumentError("delta must be > 0 for Huber weights (got delta=$delta)."))
    end

    δ = Float64(delta)

    # Allocate the weights matrix once and fill it in place
    Ω = Matrix{Float64}(undef, size(R))

    # Fill weights entry-wise
    @inbounds for j in axes(R, 2), i in axes(R, 1)
        # Residual magnitude at entry (i, j)
        ar = abs(Float64(R[i, j]))

        if ar <= δ
            # Quadratic region: full weight
            Ω[i, j] = 1.0
        else
            # Linear region: downweight large residuals
            Ω[i, j] = δ / (ar + ϵ)
        end
    end

    return Ω
end


"""
    update_huber(X, W, H, Ω; ϵ=eps(Float64))

Perform **one weighted multiplicative update step** for robust NMF with Huber IRLS weights.

We interpret Huber as a weighted least-squares problem at each IRLS step:

    min{W,H ≥ 0} ||Ω ⊙ (X - W*H)||_F^2

Given Ω (same size as X), the standard Frobenius multiplicative updates become:
    H ← H ⊙ (W' * (Ω ⊙ X)) ./ (W' * (Ω ⊙ (W*H)) + ϵ)
    W ← W ⊙ ((Ω ⊙ X) * H') ./ ((Ω ⊙ (W*H)) * H' + ϵ)

Returns updated (W, H).
"""
function update_huber(
    X::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    H::AbstractMatrix{<:Real},
    Ω::AbstractMatrix{<:Real};
    ϵ::Real = eps(Float64))
    
    # Convert epsilon once
    eps64 = Float64(ϵ)

    # Compute the current reconstruction once
    WH = W * H

    # Apply weights to X and WH (element-wise)
    # These are the weighted "data" and weighted "model" for the update rules
    ΩX = Ω .* X
    ΩWH = Ω .* WH

    # --- Update H ---
    # Numerator: W' * (Ω ⊙ X)
    numH = W' * ΩX

    # Denominator: W' * (Ω ⊙ (W*H)) + ϵ
    denH = W' * ΩWH .+ eps64

    # Multiplicative update (element-wise)
    H .= H .* (numH ./ denH)

    # Recompute WH after updating H (keeps the next step consistent)
    WH = W * H
    ΩWH = Ω .* WH

    # --- update W ---
    # Numerator: (Ω ⊙ X) * H'
    numW = ΩX * H'

    # Denominator: (Ω ⊙ (W*H)) * H' + ϵ
    denW = ΩWH * H' .+ eps64

    # Multiplicative update (element-wise)
    W .= W .* (numW ./ denW)

    return W, H
end


"""
    robustnmf(X; kwargs...)

Default robust NMF entry point.

This calls **Huber-loss robust NMF** implementation by default.
"""
robustnmf(X::AbstractMatrix{<:Real}; kwargs...) = robustnmf_huber(X; kwargs...)