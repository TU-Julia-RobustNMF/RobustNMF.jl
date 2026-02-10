using Plots
using Measures
using LinearAlgebra

"""
    plot_basis_vectors(W::AbstractMatrix; img_shape=nothing, max_components::Int=16, 
                      title::String="Basis Vectors (W)", layout=nothing)

Visualize the basis vectors (columns of W) as heatmaps or images.

# Arguments
- `W::AbstractMatrix`: Basis matrix of size (m, rank).

# Keyword Arguments
- `img_shape`: Tuple `(height, width)` to reshape each basis vector as an image. 
               If `nothing`, displays as 1D heatmaps.
- `max_components::Int=16`: Maximum number of components to display.
- `title::String`: Plot title.
- `layout`: Custom layout tuple (rows, cols). If `nothing`, auto-computed.

# Returns
- A `Plots.Plot` object showing the basis vectors.

# Examples
```julia
W, H, _ = nmf(X; rank=10)
plot_basis_vectors(W; max_components=10)

# For image data with known dimensions
plot_basis_vectors(W; img_shape=(28, 28), max_components=16)
```
"""
function plot_basis_vectors(W::AbstractMatrix; img_shape=nothing, max_components::Int=16,
                            title::String="Basis Vectors (W)", layout=nothing)
    
    m, rank = size(W)
    n_display = min(rank, max_components)
    
    # Auto-compute layout if not provided
    if layout === nothing
        ncols = Int(ceil(sqrt(n_display)))
        nrows = Int(ceil(n_display / ncols))
        layout = (nrows, ncols)
    end
    
    plots = Vector{Plots.Plot}(undef, n_display)
    
    for i in 1:n_display
        basis = @view W[:, i]
        
        if img_shape !== nothing
            # Reshape as image
            h, w = img_shape
            if length(basis) != h * w
                error("img_shape dimensions don't match basis vector length")
            end
            img = reshape(basis, h, w)
            p = heatmap(img, aspect_ratio=:equal, axis=nothing, border=:none,
                       c=:grays, title="W$i", titlefontsize=8, colorbar=false, yflip=true)
        else
            # Display as 1D heatmap
            p = heatmap(reshape(basis, :, 1), aspect_ratio=:auto, 
                       c=:viridis, title="W$i", titlefontsize=8,
                       yaxis=false, xaxis=false, colorbar=false)
        end
        
        push!(plots, p)
    end
    
    # Fill remaining subplots with empty plots if needed
    total_slots = layout[1] * layout[2]
    while length(plots) < total_slots
        push!(plots, plot(framestyle=:none))
    end
    
    plot(plots..., layout=layout, plot_title=title, size=(800, 600))
end


"""
    plot_activation_coefficients(H::AbstractMatrix; max_samples::Int=10, 
                                 title::String="Activation Coefficients (H)")

Visualize the activation coefficient matrix H as a heatmap or as individual sample profiles.

# Arguments
- `H::AbstractMatrix`: Coefficient matrix of size (rank, n).

# Keyword Arguments
- `max_samples::Int=10`: Maximum number of samples to display (if showing individual profiles).
- `title::String`: Plot title.

# Returns
- A `Plots.Plot` object.

# Examples
```julia
W, H, _ = nmf(X; rank=10)
plot_activation_coefficients(H)
```
"""
function plot_activation_coefficients(H::AbstractMatrix; max_samples::Int=10,
                                      title::String="Activation Coefficients (H)")
    
    rank, n = size(H)
    
    # If H is small enough, show full heatmap
    if n <= 100 && rank <= 50
        p = heatmap(H, xlabel="Samples", ylabel="Components", 
                   title=title, c=:viridis, aspect_ratio=:auto)
        return p
    end
    
    # Otherwise, show individual sample profiles
    n_display = min(n, max_samples)
    plots = Plots.Plot[]
    
    for i in 1:n_display
        p = bar(H[:, i], xlabel="Component", ylabel="Activation",
               title="Sample $i", legend=false, color=:steelblue)
        push!(plots, p)
    end
    
    ncols = Int(ceil(sqrt(n_display)))
    nrows = Int(ceil(n_display / ncols))
    
    plot(plots..., layout=(nrows, ncols), plot_title=title, size=(800, 600), margin=5mm)
end


"""
    plot_reconstruction_comparison(X_original::AbstractMatrix, X_recon::AbstractMatrix;
                                   img_shape=nothing, n_samples::Int=5,
                                   title::String="Reconstruction Comparison")

Compare original data with reconstructed data side by side.

# Arguments
- `X_original::AbstractMatrix`: Original data matrix.
- `X_recon::AbstractMatrix`: Reconstructed data matrix (W * H).

# Keyword Arguments
- `img_shape`: Tuple `(height, width)` for reshaping columns as images.
- `n_samples::Int=5`: Number of samples to display.
- `title::String`: Plot title.

# Returns
- A `Plots.Plot` object showing original vs reconstructed samples.

# Examples
```julia
W, H, _ = nmf(X; rank=10)
X_recon = W * H
plot_reconstruction_comparison(X, X_recon; img_shape=(28, 28), n_samples=6)
```
"""
function plot_reconstruction_comparison(X_original::AbstractMatrix, X_recon::AbstractMatrix;
                                       img_shape=nothing, n_samples::Int=5,
                                       title::String="Reconstruction Comparison")
    
    m, n = size(X_original)
    @assert size(X_recon) == (m, n) "X_original and X_recon must have same dimensions"
    
    n_display = min(n, n_samples)
    plots = Plots.Plot[]
    
    for i in 1:n_display
        orig = X_original[:, i]
        recon = X_recon[:, i]
        
        if img_shape !== nothing
            h, w = img_shape
            orig_img = reshape(orig, h, w)
            recon_img = reshape(recon, h, w)
            
            p1 = heatmap(orig_img, aspect_ratio=:equal, axis=nothing, border=:none,
                        c=:grays, title="Original $i", titlefontsize=8, colorbar=false, yflip=true)
            p2 = heatmap(recon_img, aspect_ratio=:equal, axis=nothing, border=:none,
                        c=:grays, title="Recon $i", titlefontsize=8, colorbar=false, yflip=true)
            
            push!(plots, p1, p2)
        else
            p = plot(orig, label="Original", lw=2)
            plot!(p, recon, label="Reconstructed", lw=2, linestyle=:dash)
            plot!(p, xlabel="Feature", ylabel="Value", title="Sample $i")
            push!(plots, p)
        end
    end
    
    if img_shape !== nothing
        ncols = 2 * n_display
        layout = (2, n_display)
    else
        ncols = Int(ceil(sqrt(n_display)))
        nrows = Int(ceil(n_display / ncols))
        layout = (nrows, ncols)
    end
    
    plot(plots..., layout=layout, plot_title=title, size=(1000, 400), margin=7mm)
end


"""
    plot_convergence(history::Vector;
                     title::String="NMF Convergence",
                     objective::Symbol=:auto,
                     ylabel::Union{Nothing,String}=nothing, 
                     log_scale::Bool=true)

Plot the convergence history (`history`) produced by an NMF routine.

This function is intentionally objective-agnostic: depending on the algorithm,
`history` may contain Frobenius reconstruction error (standard NMF), Huber loss
(robust NMF), or another objective value.

# Arguments
- `history::Vector`: Objective values recorded per iteration.

# Keyword Arguments
- `title::String`: Plot title.
- `objective::Symbol=:auto`: Hint for labeling the objective.
    - `:frobenius` → "Frobenius Error"
    - `:huber`     → "Huber Loss"
    - `:l21`       → "L2,1 Loss"
    - `:auto`      → "Objective" (neutral default)
- `ylabel::Union{Nothing,String}=nothing`: Explicit y-axis label.
If provided, this ovverrides `objective`.
- `log_scale::Bool=true`: Use logarithmic scale for y-axis.

# Returns
- A `Plots.Plot` object.

# Examples
```julia
W, H, history = nmf(X; rank=10, maxiter=500)
plot_convergence(history; objective=:frobenius)

W, H, history = robustnmf(X; rank=10, maxiter=500)
plot_convergence(history; objective=:huber)
```
"""
function plot_convergence(
    history::Vector; 
    title::String="NMF Convergence",
    objective::Symbol=:auto,
    ylabel::Union{Nothing,String}=nothing,
    log_scale::Bool=true)

    default_ylabel = if objective === :frobenius
        "Frobenius Error"
    elseif objective === :huber
        "Huber Loss"
    elseif objective === :l21
        "L2,1 Loss"
    else
        "Objective"
    end

    ylab = ylabel === nothing ? default_ylabel : ylabel
    
    p = plot(
        1:length(history), 
        history, 
        xlabel="Iteration",
        ylabel=ylab,
        title=title, 
        lw=2, 
        color=:blue,
        legend=false
    )
    
    if log_scale
        plot!(p, yscale=:log10)
    end
    
    return p
end


"""
    plot_nmf_summary(X::AbstractMatrix, 
                     W::AbstractMatrix, 
                     H::AbstractMatrix, 
                     history::Vector; 
                     img_shape=nothing, 
                     title::String="NMF Summary",
                     max_basis::Int=10,
                     max_samples::Int=10,
                     objective::Symbol=:auto,
                     convergence_ylabel::Union{Nothing,String}=nothing)

Create a summary visualization for an NMF result.

The summary consists of:
1. Basis vectors / components (columns of `W`)
2. Activating coefficients (rows of `H`)
3. Reconstruction comparison (original vs reconstructed data)
4. Convergence curve (objective value over iterations)

This function is **algorithm-agnostic** and works for:
- Standard NMF (Frobenius objective)
- Robist NMF with Huber loss
- Legacy L2,1 NMF

# Arguments
- `X::AbstractMatrix`: Original non-negative data matrix `(m × n)`.
- `W::AbstractMatrix`: Basis matrix `(m × r)`.
- `H::AbstractMatrix`: Coefficient matrix `(r × n)`.
- `history::Vector`: Objective values recorded during optimization.

# Keyword Arguments
- `img_shape`: Optional tuple `(height, width)` if columns of `X` or `W` represent vectorized images.
- `title`: Overalltitle for the summary figure.
- `max_basis::Int=10`: Maximum number of basis vectors (columns of `W`) to visualize.
- `max_samples::Int=10`: Maximum number of samples / activations to visualize.
- `objective`: Type of objective used to generate `history`.
    - `:frobenius` → squared Frobenius reconstruction objective
    - `:huber`     → Huber loss (robust NMF)
    - `:l21`       → L2,1 Loss
    - `:auto`      → neutral label ("Objective")
- `convergence_ylabel`: Optional explicit y-axis label for the convergence plot.
  If provided, this overrides the label implied by `objective`.

# Returns
- A `Plots.Plot` object with a comprehensive summary.

# Examples
```julia
# --- Standard NMF ---
W, H, history = nmf(X; rank=10, maxiter=500)
plot_nmf_summary(
    X, W, H, history;
    objective=:frobenius
    img_shape=(28, 28)
)

# --- Robust NMF (Huber loss) ---
W, H, history = robustnmf(X; rank=10, maxiter=500, delta=1.0)
plot_nmf_summary(
    X, W, H, history;
    objective=:huber
    img_shape=(28, 28)
)

# --- Neutral / algorithm-agnostic usage ---
plot_nmf_summary(X, W, H, history)
```
"""
function plot_nmf_summary(
    X::AbstractMatrix, 
    W::AbstractMatrix, 
    H::AbstractMatrix,
    history::Vector; 
    img_shape=nothing,
    max_basis::Int=4,
    max_samples::Int=4,
    objective::Symbol=:auto,
    convergence_ylabel::Union{Nothing,String}=nothing,
    title::String="NMF Summary"
)
    # ================================
    # Create the four main sub-figures
    # ================================

    # --- Basis vectors (columns of W) ---
    p1 = plot_basis_vectors(
        W; 
        img_shape=img_shape, 
        max_components=max_basis, 
        title="Basis Vectors (W)"
    )
    
    # --- Activating coefficients (rows of H) ---
    p2 = plot_activation_coefficients(
        H;
        max_samples=max_samples,
        title="Activating Coefficients (H)"
    )

    # --- Reconstruction comparison (original vs reconstructed data) ---
    X_recon = W * H
    p3 = plot_reconstruction_comparison(
        X, X_recon;
        img_shape=img_shape,
        n_samples=max_samples,
        title="Reconstruction"
    )
    
    # --- Convergence history (objective over iterations) ---
    p4 = plot_convergence(
        history;
        title="Convergence",
        objective=objective,
        ylabel=convergence_ylabel
    )


    # =========================
    # Info panel: extra metrics
    # =========================
    # (1) Frobenius reconstruction error: baseline "distane" between X and WH.
    # It is the most standard reconstruction metric and is comparable across runs, even if the
    # optimization objective is different (e.g. Huber, L2,1).
    fro_err = norm(X - X_recon)  # ‖X - WH‖_F

    # (2) Relative Frobenius error: scale-normalized "distance" between X and WH.
    # It makes runs on different datasets / scalings comparable (percent-like).
    rel_fro_err = fro_err / (norm(X) + eps(Float64))

    # (3) Explained energy (pseudo-R²): 1 - ‖X-WH‖²/‖X‖².
    # Tells how much of X is captured by WH and helps compare ranks.
    fro2 = fro_err^2
    xfro2 = norm(X)^2 + eps(Float64)
    explained_energy = 1 - fro2 / xfro2

    # (4) Sparsity of W and H (fraction near zero).
    # Sparsity is a key interpretability aspect of NMF ("parts-based" structure).
    # A simple threshold-based sparsity is easy to interpret and good for quick diagnostics.
    sparsity_threshold = 1e-8
    sparsity_W = sum(W .<= sparsity_threshold) / length(W)
    sparsity_H = sum(H .<= sparsity_threshold) / length(H)

    # (5) Active components: number of rows of H that are "used" (non-trivial norm).
    # It reveals if some components are effectively dead -> rank might be too high or 
    # optimization got stuck.
    rank = size(H, 1)
    active_threshold = 1e-8
    active_components = sum(norm(@view(H[i, :])) > active_threshold for i in 1:rank)

    # (6) Objective improvement percentage (from first to last history value).
    # A quick convergence sanity check: did we meaningfully reduce the objective?
    initial_obj = isempty(history) ? NaN : history[1]
    final_obj   = isempty(history) ? NaN : history[end]
    obj_drop_pct = (!isnan(initial_obj) && initial_obj != 0) ? (initial_obj - final_obj) / initial_obj : NaN


    # Determine objective label for history (may be Frobenius², Huber, or L2,1)
    obj_label = if objective === :frobenius
        "Objective (Squared Frobenius)"
    elseif objective === :huber
        "Objective (Huber loss)"
    elseif objective === :l21
        "Objective (L2,1 loss)"
    else
        "Objective"
    end

    # Formal helper for percentages
    pct = (x; digits=2) -> round(100 * x, digits=digits)

    # --- Info panel text ---
    info_text = """
    $(title)
    --------------------------------
    
    Dataset & Model:
    ----------------

    Data size                : $(size(X))

    Rank                     : $(size(W, 2))

    Iterations               : $(length(history))
    

    Reconstruction Quality:
    -----------------------

    Frobenius Error          : $(round(fro_err, digits=6))

    Relative Error           : $(pct(rel_fro_err, digits=2))%

    Explained Energy         : $(pct(explained_energy, digits=2))%


    Factor Structure:
    -----------------

    Sparsity(W)              : $(pct(sparsity_W, digits=2))%

    Sparsity(H)              : $(pct(sparsity_H, digits=2))%

    Active Components        : $(active_components) / $(rank)


    Optimization:
    -------------

    $(obj_label) : $(round(final_obj, digits=6))
    
    Objective Reduction      : $(isnan(obj_drop_pct) ? "n/a" : string(pct(obj_drop_pct, digits=2), "%"))
    """

    p_info = plot(framestyle=:none, showaxis=false, ticks=false)
    n_lines = count(==('\n'), info_text) + 1
    y_start = min(0.98, 0.98 - 0.010 * n_lines)
    annotate!(p_info, 0.03, y_start, text(info_text, :left, 10, :courier))
    
    # --- Combine plots including the info panel ---
    # Layout with five panels:
    #   left:   basis (top), activations (bottom)
    #   middle: reconstruction (top), convergence (bottom)
    #   right:  info panel

    l = @layout [[a{0.60h}; b{0.35h}] [c{0.60h}; d{0.35h}] e{0.35w}]

    return plot(
        p1, p2, p3, p4, p_info;
        layout=l,
        size=(2000, 1000),
        left_margin=10mm
    )
end


"""
    plot_image_reconstruction(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix,
                             img_shape::Tuple{Int,Int}; indices=nothing, n_images::Int=5)

Specialized function for visualizing image reconstruction quality.
Shows original, reconstructed, and difference images side by side.

# Arguments
- `X::AbstractMatrix`: Original image data (each column is a flattened image).
- `W::AbstractMatrix`: Basis matrix.
- `H::AbstractMatrix`: Coefficient matrix.
- `img_shape::Tuple{Int,Int}`: Image dimensions (height, width).

# Keyword Arguments
- `indices`: Specific image indices to display. If `nothing`, randomly selected.
- `n_images::Int=5`: Number of images to display.

# Returns
- A `Plots.Plot` object.

# Examples
```julia
W, H, _ = nmf(X; rank=20)
plot_image_reconstruction(X, W, H, (64, 64); n_images=6)
```
"""
function plot_image_reconstruction(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix,
                                  img_shape::Tuple{Int,Int}; indices=nothing, n_images::Int=5)
    
    h, w = img_shape
    m, n = size(X)
    X_recon = W * H
    
    # Select indices
    if indices === nothing
        indices = rand(1:n, min(n_images, n))
    else
        indices = indices[1:min(length(indices), n_images)]
    end
    
    plots = Plots.Plot[]
    
    for idx in indices
        # Original
        orig_img = reshape(X[:, idx], h, w)
        p1 = heatmap(orig_img, aspect_ratio=:equal, axis=nothing, border=:none,
                    c=:grays, title="Original", titlefontsize=8, colorbar=false)
        
        # Reconstructed
        recon_img = reshape(X_recon[:, idx], h, w)
        p2 = heatmap(recon_img, aspect_ratio=:equal, axis=nothing, border=:none,
                    c=:grays, title="Reconstructed", titlefontsize=8, colorbar=false)
        
        # Difference (error)
        diff_img = abs.(orig_img - recon_img)
        p3 = heatmap(diff_img, aspect_ratio=:equal, axis=nothing, border=:none,
                    c=:reds, title="|Error|", titlefontsize=8, colorbar=false)
        
        push!(plots, p1, p2, p3)
    end
    
    nrows = length(indices)
    layout = (nrows, 3)
    
    plot(plots..., layout=layout, 
         plot_title="Image Reconstruction Quality",
         size=(900, 300 * nrows))
end
