using LinearAlgebra
using Random
using Statistics
# L2,1-NMF
function robustnmf(X; rank::Int = 10, maxiter::Int = 500, tol::Float64 = 1e-4)
  m, n = size(X)
        F = rand(m, rank)
        G = rand(rank, n)
        @assert minimum(X) >= 0 "X must be non-negative"

  # update until convergence is reached or at most maxiter times
  for iter = 1:maxiter
    F,G = update(X,F,G)

    # compute error
    error = l21norm(X - F * G)

    # break loop if sufficiently converged
    if (error < tol)
      return F,G
    else
      continue
    end
  end
  return F,G
end

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
    
    # Ensure non-negativity
    @. G_new = max(G_new, eps_update)
    
    return F_new, G_new
end


"""
    l21_nmf(X; rank=10, maxiter=500, tol=1e-4, seed=nothing)

L2,1-Norm Regularized Non-negative Matrix Factorization.

Minimizes: ||X - FG||_{2,1} where the L2,1-norm promotes robustness
to sample-wise outliers (entire corrupted columns in X).

# Arguments
- `X::AbstractMatrix{<:Real}`: Non-negative data matrix (m × n)

# Keyword Arguments
- `rank::Int=10`: Number of latent components
- `maxiter::Int=500`: Maximum iterations
- `tol::Float64=1e-4`: Convergence tolerance (absolute error threshold)
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility

# Returns
- `F::Matrix{Float64}`: Basis matrix (m × rank)
- `G::Matrix{Float64}`: Coefficient matrix (rank × n)
- `history::Vector{Float64}`: L2,1-norm error at each iteration

# Examples
```julia
X = rand(50, 30)
F, G, hist = l21_nmf(X; rank=5, maxiter=200)
```
"""
function l21_nmf(X::AbstractMatrix{<:Real}; 
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
        F, G = l21_update(X, F, G)
        
        # Compute L2,1-norm error
        error = l21norm(X - F * G)
        history[iter] = error
        
        # Check convergence
        if error < tol
            history = history[1:iter]
            break
        end
    end
    
    return F, G, history
end
