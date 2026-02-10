using LinearAlgebra: norm
using Random: Random, MersenneTwister, rand

# --- Helper Functions ---
"""
    huber_loss(R::AbstractMatrix{<:Real}, delta::Real; ϵ=eps(Float64))

Compute the **Huber loss** for a residual matrix `R`.

Huber loss is robust to outliers:
- For small residuals it behaves like squared error (L2).
- For large residuals it behaves like absolute error (L1), reducing the impact of outliers.

Element-wise definition for residual `r`:
- if |r| ≤ δ: 0.5 * r^2
- else:       δ * (|r| - 0.5*δ)

Returns the **sum** over all entries of `R`.

# Arguments
- `R::AbstractMatrix{<:Real}`: Residual matrix.
- `delta::Real`: Huber threshold δ (> 0).

# Keyword Arguments
- `ϵ=eps(Float64)`: Small constant for numerical stability (used in validation/consistency).

# Returns
- `loss::Float64`: Sum of Huber losses over all entries of `R`.

# Side Effects
- None.

# Errors
- `ArgumentError`: If `delta <= 0`.

# Notes
- For `|r| ≤ δ`, the loss is quadratic; for `|r| > δ`, it is linear.
- Summation uses the element type of `R` for type stability.

"""
function huber_loss(R::AbstractMatrix{<:Real}, delta::Real; ϵ::Real = eps(Float64))
    # Basic parameter validation:
    # delta controls where we transition from quadratic (L2) to linear (L1-like).
    if delta <= 0
        throw(ArgumentError("delta must be > 0 for Huber loss (got delta=$delta)."))
    end

    # Convert delta and work with element type of R
    T = eltype(R)
    δ = convert(T, delta)

    # Accumulate total loss
    total = zero(T)

    # Loop explicitly for performance and to avoid temporary allocations
    @inbounds for r in R
        # Residual magnitude
        ar = abs(r)

        if ar <= δ
            # Quadratic region: 0.5 * r^2
            total += T(0.5) * ar * ar
        else
            # Linear region: δ*(|r| - 0.5*δ)
            total += δ * (ar - T(0.5) * δ)
        end
    end

    return total
end


"""
    huber_weights(R::AbstractMatrix{<:Real}, delta::Real; ϵ=eps(Float64))

Compute the **Huber IRLS weights** matrix `Ω` for a residual matrix `R`.

We use an iteratively reweighted least squares (IRLS) interpretation:
- Small residuals get weight 1.0 (quadratic region).
- Large residuals get weight δ / (|r| + ϵ), which downweights outliers.

Element-wise:
- if |r| ≤ δ: w = 1
- else:       w = δ / (|r| + ϵ)

Returns `Ω` with the same size as `R`.

# Arguments
- `R::AbstractMatrix{<:Real}`: Residual matrix.
- `delta::Real`: Huber threshold δ (> 0).

# Keyword Arguments
- `ϵ=eps(Float64)`: Small constant to avoid division by zero.

# Returns
- `Ω::AbstractMatrix`: Weight matrix of same size as `R` (same type as `R`).

# Side Effects
- None.

# Errors
- `ArgumentError`: If `delta <= 0`.

# Notes
- Weights are `1.0` in the quadratic region and decrease for large residuals.
- Used to build weighted multiplicative updates in IRLS.

"""
function huber_weights(R::AbstractMatrix{<:Real}, delta::Real; ϵ::Real = eps(Float64))
    # Validate delta: must be positive to define a Huber threshold.
    if delta <= 0
        throw(ArgumentError("delta must be > 0 for Huber weights (got delta=$delta)."))
    end

    T = eltype(R)
    δ = convert(T, delta)

    # Allocate the weights matrix once and fill it in place
    Ω = similar(R)

    # Fill weights entry-wise
    @inbounds for j in axes(R, 2), i in axes(R, 1)
        # Residual magnitude at entry (i, j)
        ar = abs(R[i, j])

        if ar <= δ
            # Quadratic region: full weight
            Ω[i, j] = one(T)
        else
            # Linear region: downweight large residuals
            Ω[i, j] = δ / (ar + ϵ)
        end
    end

    return Ω
end


"""
    huber_weights!(Ω::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real}, delta::Real; ϵ=eps(Float64))

In-place version of `huber_weights`. Writes weights into preallocated `Ω`.

# Arguments
- `Ω::AbstractMatrix{<:Real}`: Preallocated weight matrix (same size as `R`).
- `R::AbstractMatrix{<:Real}`: Residual matrix.
- `delta::Real`: Huber threshold δ (> 0).

# Keyword Arguments
- `ϵ=eps(Float64)`: Small constant to avoid division by zero.

# Returns
- `Ω::AbstractMatrix`: The filled weights matrix.

# Errors
- `ArgumentError`: If `delta <= 0` or size mismatch.
"""
function huber_weights!(Ω::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real},
                        delta::Real; ϵ::Real = eps(Float64))
    if delta <= 0
        throw(ArgumentError("delta must be > 0 for Huber weights (got delta=$delta)."))
    end
    if size(Ω) != size(R)
        throw(ArgumentError("Ω must be the same size as R (got size(Ω)=$(size(Ω)), size(R)=$(size(R)))."))
    end

    T = eltype(R)
    δ = convert(T, delta)

    @inbounds for j in axes(R, 2), i in axes(R, 1)
        ar = abs(R[i, j])
        if ar <= δ
            Ω[i, j] = one(T)
        else
            Ω[i, j] = δ / (ar + ϵ)
        end
    end

    return Ω
end


"""
    l21_loss(X::AbstractMatrix)

Compute the L2,1-norm of matrix `X`.
The L2,1-norm is the sum of the L2-norms of each column.

# Arguments
- `X::AbstractMatrix`: Input matrix.

# Keyword Arguments
- None.

# Returns
- `value::Float64`: Sum of L2-norms of columns.

# Side Effects
- None.

# Errors
- None.

# Notes
- Legacy helper used by L2,1-NMF.

# Examples
```jldoctest
julia> X = [1.0 2.0; 3.0 4.0];

julia> l21_loss(X) >= 0
true
```
"""
function l21_loss(X::AbstractMatrix)
    s = zero(eltype(X))
    @inbounds for col in eachcol(X)
        s += norm(col)
    end
    return s
end



# --- Update Rules ---
"""
    update_huber(X, W, H, Ω; ϵ=eps(Float64))

Perform **one weighted multiplicative update step** for robust NMF with Huber IRLS weights.

We interpret Huber as a weighted least-squares problem at each IRLS step:

    min{W,H ≥ 0} ||Ω ⊙ (X - W*H)||_F^2

Given Ω (same size as X), the standard Frobenius multiplicative updates become:
    H ← H ⊙ (W' * (Ω ⊙ X)) ./ (W' * (Ω ⊙ (W*H)) + ϵ)
    W ← W ⊙ ((Ω ⊙ X) * H') ./ ((Ω ⊙ (W*H)) * H' + ϵ)

Returns updated `(W, H)`.

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix `(m, n)`.
- `W::AbstractMatrix{<:Real}`: Current basis matrix `(m, rank)`.
- `H::AbstractMatrix{<:Real}`: Current coefficient matrix `(rank, n)`.
- `Ω::AbstractMatrix{<:Real}`: IRLS weight matrix (same size as `X`).

# Keyword Arguments
- `ϵ=eps(Float64)`: Small constant for numerical stability.

# Returns
- `W::AbstractMatrix{<:Real}`: Updated basis matrix.
- `H::AbstractMatrix{<:Real}`: Updated coefficient matrix.

# Side Effects
- Updates `W` and `H` in-place.

# Errors
- None.

# Notes
- Multiplicative updates preserve non-negativity if `W` and `H` start non-negative.
- Uses `Ω ⊙ X` and `Ω ⊙ (W*H)` to compute weighted updates.

"""
function update_huber(
    X::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    H::AbstractMatrix{<:Real},
    Ω::AbstractMatrix{<:Real};
    ϵ::Real = eps(Float64))

    # Convert epsilon to match the element type for numerical stability
    T = eltype(X)
    ϵ_T = convert(T, ϵ)  # ϵ in type T for denominators

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
    denH = W' * ΩWH .+ ϵ_T

    # Multiplicative update (element-wise)
    H .= H .* (numH ./ denH)

    # Recompute WH after updating H (keeps the next step consistent)
    WH = W * H
    ΩWH = Ω .* WH

    # --- update W ---
    # Numerator: (Ω ⊙ X) * H'
    numW = ΩX * H'

    # Denominator: (Ω ⊙ (W*H)) * H' + ϵ
    denW = ΩWH * H' .+ ϵ_T

    # Multiplicative update (element-wise)
    W .= W .* (numW ./ denW)

    return W, H
end


"""
    update_l21(X, W, H; eps_update=1e-10)

Perform one iteration of L2,1-NMF multiplicative updates.

The L2,1-norm promotes row sparsity in the residual matrix, making the
algorithm robust to sample-wise (column-wise) outliers.

# Arguments
- `X::AbstractMatrix`: Data matrix `(m, n)`.
- `W::AbstractMatrix`: Current basis matrix `(m, rank)`.
- `H::AbstractMatrix`: Current coefficient matrix `(rank, n)`.

# Keyword Arguments
- `eps_update::Real=1e-10`: Small constant for numerical stability.

# Returns
- `W_new::Matrix{Float64}`: Updated basis matrix.
- `H_new::Matrix{Float64}`: Updated coefficient matrix.

# Side Effects
- None.

# Errors
- None.

# Notes
- Legacy algorithm kept for compatibility; Huber is the default robust method.
- Update uses a diagonal reweighting matrix `D` derived from column residuals.

"""
struct L21Workspace{T}
    WH::Matrix{T}
    R::Matrix{T}
    d::Vector{T}
    XD::Matrix{T}
    W1::Matrix{T}
    W2::Matrix{T}
    WtW::Matrix{T}
    H1::Matrix{T}
    H2::Matrix{T}
end

function L21Workspace(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix)
    T = eltype(X)
    m, n = size(X)
    rank = size(W, 2)
    return L21Workspace{T}(
        similar(X, m, n),   # WH
        similar(X, m, n),   # R
        Vector{T}(undef, n),
        similar(X, m, n),   # XD
        similar(W, m, rank),
        similar(W, m, rank),
        similar(W, rank, rank),
        similar(H, rank, n),
        similar(H, rank, n),
    )
end

function update_l21!(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix,
                     ws::L21Workspace;
                     eps_update::Real=1e-10)

    T = eltype(X)
    epsT = T(eps_update)

    # Compute residual once: R = X - W*H
    mul!(ws.WH, W, H)
    @. ws.R = X - ws.WH

    # d[i] = 1 / (2 * ||r_i||_2 + eps)
    @inbounds for (i, col) in enumerate(eachcol(ws.R))
        ws.d[i] = inv(T(2) * norm(col) + epsT)
    end

    # Pre-scale columns instead of forming Diagonal(d)
    drow = reshape(ws.d, 1, :)
    @. ws.XD = X .* drow

    # Update W using multiplicative update rule
    mul!(ws.W1, ws.XD, H')
    @. ws.R = ws.WH .* drow
    mul!(ws.W2, ws.R, H')
    @. ws.W2 = ws.W2 + epsT
    @. W = max(W .* (ws.W1 ./ ws.W2), epsT)

    # Update H using multiplicative update rule
    mul!(ws.H1, W', ws.XD)
    mul!(ws.WtW, W', W)
    mul!(ws.H2, ws.WtW, H)
    @. ws.H2 = ws.H2 .* drow
    @. ws.H2 = ws.H2 + epsT
    @. H = max(H .* (ws.H1 ./ ws.H2), epsT)

    return W, H
end

function update_l21(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix;
                    eps_update::Real=1e-10)
    ws = L21Workspace(X, W, H)
    Wc = copy(W)
    Hc = copy(H)
    return update_l21!(X, Wc, Hc, ws; eps_update=eps_update)
end

# --- Full Algorithms ---
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

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix of size `(m, n)`.

# Keyword Arguments
- `rank::Int=10`: Factorization rank.
- `maxiter::Int=500`: Maximum number of iterations.
- `tol::Real=1e-4`: Relative tolerance for stopping based on objective change.
- `delta::Real=1.0`: Huber threshold δ (must be > 0).
- `seed=nothing`: Optional random seed for reproducibility.

# Returns
- `W::Matrix{Float64}`: Non-negative basis matrix of size `(m, rank)`.
- `H::Matrix{Float64}`: Non-negative coefficient matrix of size `(rank, n)`.
- `history::Vector{Float64}`: Huber objective values per iteration.

# Side Effects
- None. (The function does not modify `X`.)

# Errors
- `ArgumentError`: If `X` contains negative entries or parameters are invalid.

# Notes
- Uses IRLS weights via `huber_weights` and weighted multiplicative updates.
- Convergence check is based on relative change of Huber objective.
- Initialization uses a local RNG seeded by `seed` when provided.

# Examples
```jldoctest
julia> using RobustNMF

julia> X, _, _ = generate_synthetic_data(30, 20; rank=5, seed=123);

julia> W, H, history = robustnmf_huber(X; rank=5, maxiter=100, tol=1e-6, delta=1.0, seed=123);

julia> size(W), size(H), length(history) > 0
((30, 5), (5, 20), true)
```
"""
function robustnmf_huber(
    X::AbstractMatrix{<:Real};
    rank::Int = 10,
    maxiter::Int = 500,
    tol::Real = 1e-4,
    delta::Real = 1.0,
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
    T = typeof(one(eltype(X)))

    # Initialize W and H with random non-negative values
    W = rand(rng, T, m, rank)
    H = rand(rng, T, rank, n)

    # Small constant to avoid division by zero in multiplicative updates
    ϵ = eps(T)

    # Track objective history (Huber loss values)
    history = T[]
    prev_obj = T(Inf)
    Ω = similar(X)

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
        huber_weights!(Ω, R, delta; ϵ=ϵ)

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
    robustnmf_l21(X; rank=10, maxiter=500, tol=1e-4, seed=nothing)

L2,1-Norm Regularized Non-negative Matrix Factorization.

Minimizes: ||X - WH||_{2,1} where the L2,1-norm promotes robustness
to sample-wise outliers (entire corrupted columns in `X`).

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix `(m, n)`.

# Keyword Arguments
- `rank::Int=10`: Factorization rank.
- `maxiter::Int=500`: Maximum number of iterations.
- `tol::Real=1e-4`: Relative tolerance for stopping based on objective change.
- `seed=nothing`: Optional random seed for reproducibility.

# Returns
- `W::Matrix{Float64}`: Non-negative basis matrix `(m, rank)`.
- `H::Matrix{Float64}`: Non-negative coefficient matrix `(rank, n)`.
- `history::Vector{Float64}`: L2,1 objective values per iteration.

# Side Effects
- None. (The function does not modify `X`.)

# Errors
- `ArgumentError`: If `X` contains negative entries or parameters are invalid.

# Notes
- Legacy algorithm kept for compatibility; Huber is the default robust method.
- Convergence check uses relative change in objective.

# Examples
```jldoctest
julia> using RobustNMF

julia> X, _, _ = generate_synthetic_data(20, 12; rank=4, seed=7);

julia> W, H, history = robustnmf_l21(X; rank=4, maxiter=50, tol=1e-3, seed=7);

julia> size(W), size(H), length(history) > 0
((20, 4), (4, 12), true)
```
"""
function robustnmf_l21(X::AbstractMatrix{<:Real};
                 rank::Int=10,
                 maxiter::Int=500,
                 tol::Real=1e-4,
                 seed::Union{Int,Nothing}=nothing)

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

    # Reproducible initialization using a local RNG
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    
    m, n = size(X)
    T = eltype(X)

    # Initialize W and H with random non-negative values
    W = rand(rng, T, m, rank) .* T(0.5) .+ T(0.1)
    H = rand(rng, T, rank, n) .* T(0.5) .+ T(0.1)

    # Track convergence history
    history = Vector{T}(undef, maxiter)
    prev_err = T(Inf)
    epsT = eps(T)
    ws = L21Workspace(X, W, H)
    
    # Iterative updates
    for iter in 1:maxiter
        # Perform one L2,1-NMF update
        update_l21!(X, W, H, ws; eps_update=epsT)
        
        # Compute L2,1-norm error
        mul!(ws.WH, W, H)
        @. ws.R = X - ws.WH
        error = l21_loss(ws.R)
        history[iter] = error
        
        # Check convergence
        if abs(prev_err - error) / (prev_err + epsT) < tol
            history = history[1:iter]
            break
        end
        prev_err = error
    end
    return W, H, history
end


# --- Public API ---
"""
    robustnmf(X; kwargs...)

Default robust NMF entry point.

This calls **Huber-loss robust NMF** implementation by default.

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix `(m, n)`.

# Keyword Arguments
- `kwargs...`: Forwarded to `robustnmf_huber`.

# Returns
- `(W, H, history)`: See `robustnmf_huber`.

# Side Effects
- None.

# Errors
- `ArgumentError`: If inputs are invalid (see `robustnmf_huber`).

# Notes
- Public API entry point; keeps the external name stable if the default robust method changes.

# Examples
```jldoctest
julia> using RobustNMF

julia> X, _, _ = generate_synthetic_data(20, 12; rank=4, seed=7);

julia> W, H, history = robustnmf(X; rank=4, maxiter=50, tol=1e-6, delta=1.0, seed=7);

julia> size(W), size(H)
((20, 4), (4, 12))
```
"""
robustnmf(X::AbstractMatrix{<:Real}; kwargs...) = robustnmf_huber(X; kwargs...)
