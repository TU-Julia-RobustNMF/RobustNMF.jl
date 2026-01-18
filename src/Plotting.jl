using Plots

# Internal helpers
function _heatmap(A; clims=nothing, kwargs...)
    clims === nothing ? heatmap(A; kwargs...) : heatmap(A; clims=clims, kwargs...)
end

function _finalize_plot(p; savepath=nothing, display_plot=true)
    if savepath !== nothing
        dir = dirname(savepath)
        if dir != "." && !isempty(dir)
            mkpath(dir)
        end
        savefig(p, savepath)
    end
    if display_plot
        display(p)
    end
    return p
end

"""
    plot_factors(W, H; savepath=nothing, display_plot=true, heatmap_kwargs...)

Plot factor matrices `W` and `H` as heatmaps in a single figure.
"""
function plot_factors(W, H; savepath=nothing, display_plot=true, heatmap_kwargs...)
    p1 = _heatmap(W; title="W", xlabel="components", ylabel="features", colorbar=true, heatmap_kwargs...)
    p2 = _heatmap(H; title="H", xlabel="samples", ylabel="components", colorbar=true, heatmap_kwargs...)
    p = plot(p1, p2; layout=(1, 2), size=(900, 400))
    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end

"""
    plot_reconstruction(X, X_noisy, W, H; savepath=nothing, display_plot=true,
                        shared_clims=true, heatmap_kwargs...)

Plot original data `X`, noisy data `X_noisy`, and reconstruction `W*H` side by side.
"""
function plot_reconstruction(
    X,
    X_noisy,
    W,
    H;
    savepath=nothing,
    display_plot=true,
    shared_clims=true,
    heatmap_kwargs...
)
    WH = W * H
    clims = shared_clims ? extrema(vcat(vec(X), vec(X_noisy), vec(WH))) : nothing

    p1 = _heatmap(X; title="X", clims=clims, colorbar=true, heatmap_kwargs...)
    p2 = _heatmap(X_noisy; title="X_noisy", clims=clims, colorbar=true, heatmap_kwargs...)
    p3 = _heatmap(WH; title="W*H", clims=clims, colorbar=true, heatmap_kwargs...)
    p = plot(p1, p2, p3; layout=(1, 3), size=(1200, 400))
    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end

"""
    plot_convergence(history; label="error", savepath=nothing, display_plot=true)
    plot_convergence(histories; labels=nothing, savepath=nothing, display_plot=true)

Plot one or more convergence curves from error histories.
"""
function plot_convergence(history::AbstractVector; label="error", savepath=nothing, display_plot=true)
    p = plot(history; label=label, xlabel="iteration", ylabel="error")
    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end

function plot_convergence(histories::AbstractVector{<:AbstractVector}; labels=nothing, savepath=nothing, display_plot=true)
    p = plot(; xlabel="iteration", ylabel="error")
    for (i, h) in enumerate(histories)
        lbl = labels === nothing ? "run $(i)" : labels[i]
        plot!(p, h; label=lbl)
    end
    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end
