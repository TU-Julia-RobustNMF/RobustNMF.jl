using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using RobustNMF
using Plots

if !isdefined(Main, :RobustNMFExampleDatasets)
    include(joinpath(@__DIR__, "datasets.jl"))
end
const DSETS = Main.RobustNMFExampleDatasets
att_faces_root = DSETS.att_faces_root

function run_demo()
    # --- Load At&T/OR dataset (40 subjects × 10 images, subfolder s1...s40) ---
    root = att_faces_root()

    subdirs = sort(filter(d -> startswith(basename(d), "s") && isdir(d), 
                        readdir(root; join=true)))

    Xs = Matrix{Float64}[]
    names = String[]
    img_shape = nothing

    for sd in subdirs
        # Each subject folder contains 1.pgm ... 10.pgm
        X_sub, shape, fnames = load_image_folder(sd; pattern=".pgm", normalize=true)
        img_shape === nothing && (img_shape = shape)

        # Prefix filenames with subject folder to keep uniqueness
        prefix = basename(sd)
        append!(names, ["$prefix/$f" for f in fnames])
        push!(Xs, X_sub)
    end

    X = hcat(Xs...)
    @info "Loaded faces matrix" size(X) img_shape length(names)

    # --- Run Standard NMF ---
    r = 25
    W, H, hist = nmf(X; rank=r, maxiter=200, tol=1e-6, seed=1)
    @info "Standard NMF done" size(W) size(H) length(hist)

    p_std = plot_nmf_summary(
        X, W, H, hist;
        img_shape=img_shape,
        max_basis=9,
        max_samples=6,
        objective=:frobenius,
        title="AT&T Faces - Standard NMF (rank=$r)"
    )
    display(p_std)


    # --- Run Robust NMF (Huber) ---
    Wr, Hr, histr = robustnmf_huber(X; rank=r, maxiter=200, tol=1e-6, delta=1.0, seed=1)
    @info "Robust NMF done" size(Wr) size(Hr) length(histr)
    
    p_rob = plot_nmf_summary(
        X, Wr, Hr, histr;
        img_shape=img_shape,
        max_basis=9,
        max_samples=6,
        objective=:huber,
        title="AT&T Faces - Robust NMF (Huber, rank=$r)"
    )
    display(p_rob)

    # Optionally save plots
    savefig(p_std, joinpath(@__DIR__, "att_faces_standard_nmf.png"))
    savefig(p_rob, joinpath(@__DIR__, "att_faces_robust_nmf.png"))
    @info "Saved plots to examples/"

    return nothing
end

run_demo()