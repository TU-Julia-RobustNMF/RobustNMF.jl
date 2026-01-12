module Plotting

using Plots
using LinearAlgebra
using Statistics

export 
    plot_basis_vectors,
    plot_activation_coefficients,
    plot_reconstruction_comparison,
    plot_convergence,
    plot_nmf_summary,
    plot_image_reconstruction


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
    
    plots = []
    
    for i in 1:n_display
        basis = W[:, i]
        
        if img_shape !== nothing
            # Reshape as image
            h, w = img_shape
            if length(basis) != h * w
                error("img_shape dimensions don't match basis vector length")
            end
            img = reshape(basis, h, w)
            p = heatmap(img, aspect_ratio=:equal, axis=nothing, border=:none,
                       c=:grays, title="W$i", titlefontsize=8, colorbar=false)
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
    plots = []
    
    for i in 1:n_display
        p = bar(H[:, i], xlabel="Component", ylabel="Activation",
               title="Sample $i", legend=false, color=:steelblue)
        push!(plots, p)
    end
    
    ncols = Int(ceil(sqrt(n_display)))
    nrows = Int(ceil(n_display / ncols))
    
    plot(plots..., layout=(nrows, ncols), plot_title=title, size=(800, 600))
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
    plots = []
    
    for i in 1:n_display
        orig = X_original[:, i]
        recon = X_recon[:, i]
        
        if img_shape !== nothing
            h, w = img_shape
            orig_img = reshape(orig, h, w)
            recon_img = reshape(recon, h, w)
            
            p1 = heatmap(orig_img, aspect_ratio=:equal, axis=nothing, border=:none,
                        c=:grays, title="Original $i", titlefontsize=8, colorbar=false)
            p2 = heatmap(recon_img, aspect_ratio=:equal, axis=nothing, border=:none,
                        c=:grays, title="Recon $i", titlefontsize=8, colorbar=false)
            
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
    
    plot(plots..., layout=layout, plot_title=title, size=(1000, 400))
end


"""
    plot_convergence(history::Vector; title::String="NMF Convergence",
                    ylabel::String="Frobenius Error", log_scale::Bool=true)

Plot the convergence history of the NMF algorithm.

# Arguments
- `history::Vector`: Vector of error values at each iteration.

# Keyword Arguments
- `title::String`: Plot title.
- `ylabel::String`: Y-axis label.
- `log_scale::Bool=true`: Use logarithmic scale for y-axis.

# Returns
- A `Plots.Plot` object.

# Examples
```julia
W, H, history = nmf(X; rank=10, maxiter=500)
plot_convergence(history)
```
"""
function plot_convergence(history::Vector; title::String="NMF Convergence",
                         ylabel::String="Frobenius Error", log_scale::Bool=true)
    
    p = plot(1:length(history), history, 
            xlabel="Iteration", ylabel=ylabel,
            title=title, lw=2, color=:blue, legend=false)
    
    if log_scale
        plot!(p, yscale=:log10)
    end
    
    return p
end


"""
    plot_nmf_summary(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix, 
                    history::Vector; img_shape=nothing, max_basis::Int=9,
                    max_samples::Int=4)

Create a comprehensive summary plot showing basis vectors, sample reconstructions,
and convergence in a single figure.

# Arguments
- `X::AbstractMatrix`: Original data matrix.
- `W::AbstractMatrix`: Basis matrix.
- `H::AbstractMatrix`: Coefficient matrix.
- `history::Vector`: Convergence history.

# Keyword Arguments
- `img_shape`: Tuple `(height, width)` for image data.
- `max_basis::Int=9`: Maximum number of basis vectors to show.
- `max_samples::Int=4`: Maximum number of reconstruction comparisons.

# Returns
- A `Plots.Plot` object with a comprehensive summary.

# Examples
```julia
W, H, history = nmf(X; rank=10)
plot_nmf_summary(X, W, H, history; img_shape=(28, 28))
```
"""
function plot_nmf_summary(X::AbstractMatrix, W::AbstractMatrix, H::AbstractMatrix,
                         history::Vector; img_shape=nothing, max_basis::Int=9,
                         max_samples::Int=4)
    
    # Create individual plots
    p1 = plot_basis_vectors(W; img_shape=img_shape, max_components=max_basis,
                           title="Basis Vectors")
    
    X_recon = W * H
    p2 = plot_reconstruction_comparison(X, X_recon; img_shape=img_shape, 
                                       n_samples=max_samples,
                                       title="Reconstructions")
    
    p3 = plot_convergence(history; title="Convergence")
    
    # Calculate error metrics
    err = norm(X - X_recon)
    rel_err = err / norm(X)
    
    # Create info text plot
    info_text = """
    NMF Summary
    ───────────────
    Data size: $(size(X))
    Rank: $(size(W, 2))
    Iterations: $(length(history))
    
    Final Error: $(round(err, digits=4))
    Relative Error: $(round(rel_err*100, digits=2))%
    """
    
    p4 = plot(framestyle=:none, showaxis=false, ticks=false)
    annotate!(p4, 0.1, 0.5, text(info_text, :left, 10, :courier))
    
    # Combine into layout
    l = @layout [
        a{0.4h}
        [b{0.6w} [c; d]]
    ]
    
    plot(p1, p2, p3, p4, layout=l, size=(1200, 900),
         plot_title="NMF Analysis Summary")
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
    
    plots = []
    
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


end  # module Plotting
