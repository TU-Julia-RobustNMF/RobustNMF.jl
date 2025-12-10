module DataPreparation

using Random

"""
generate_synthetic_data(m, n, r; seed=42)

Generate a non-negative synthetic matrix X\_clean ∈ ℝ^{m×n} with approximate rank r.
The function samples W\_true ∈ ℝ^{m×r} and H\_true ∈ ℝ^{r×n} from Uniform(0, 1) and returns:

    X_clean = W_true * H_true

Arguments
- m::Int    Number of rows of X_clean.
- n::Int    Number of columns of X_clean.
- r::Int    Target rank of the factorization.

Keyword Arguments
- seed::Integer = 42  
  Random seed for reproducible results.

Returns
(X_clean, W_true, H_true)

"""
function generate_synthetic_data(m::Int, n::Int, r::Int; seed::Integer = 42)
    rng = MersenneTwister(seed)
    W_true = rand(rng, m, r)
    H_true = rand(rng, r, n)
    X_clean = W_true * H_true
    return X_clean, W_true, H_true
end

"""
    add_noise_and_outliers(X_clean;
                           noise_std = 0.0,
                           outlier_fraction = 0.0,
                           outlier_scale = 10.0,
                           seed = 42)

Add Gaussian noise and large positive outliers to `X_clean`.
Returns a perturbed and non negative matrix.
"""
function add_noise_and_outliers(X_clean;
                                noise_std::Float64 = 0.0,
                                outlier_fraction::Float64 = 0.0,
                                outlier_scale::Float64 = 10.0,
                                seed::Integer = 42)

    rng = MersenneTwister(seed)
    X_noisy = copy(X_clean)

    # Add Gaussian noise
    if noise_std > 0
        X_noisy .+= noise_std .* randn(rng, size(X_noisy))
    end

    # Clip negative entries to zero
    X_noisy .= max.(X_noisy, 0.0)

    # Add outliers
    if outlier_fraction > 0
        m, n = size(X_noisy)
        total_entries = m * n
        k = round(Int, outlier_fraction * total_entries)
        if k > 0
            max_val = maximum(X_noisy)
            for _ in 1:k
                i = rand(rng, 1:m)
                j = rand(rng, 1:n)
                X_noisy[i, j] = max_val * outlier_scale
            end
        end
    end

    return X_noisy
end

"""
    normalize_data(X; clip_at_zero = true, mode = :none)

Normalize matrix `X` and return a new matrix.

Modes:
    :none        no scaling
    :global_max  divide by global maximum
    :column_max  divide each column by its maximum
"""
function normalize_data(X;
                        clip_at_zero::Bool = true,
                        mode::Symbol = :none)

    Xn = copy(X)

    if clip_at_zero
        Xn .= max.(Xn, 0.0)
    end

    if mode == :none
        return Xn
    elseif mode == :global_max
        max_val = maximum(Xn)
        if max_val > 0
            Xn ./= max_val
        end
        return Xn
    elseif mode == :column_max
        m, n = size(Xn)
        for j in 1:n
            col_max = maximum(@view Xn[:, j])
            if col_max > 0
                @views Xn[:, j] ./= col_max
            end
        end
        return Xn
    else
        error("Unknown normalization mode: $mode")
    end
end

end # module
