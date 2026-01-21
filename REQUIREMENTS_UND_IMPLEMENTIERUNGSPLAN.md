# RobustNMF.jl - Anforderungsanalyse und Implementierungsplan

## Dokument-Übersicht

Dieses Dokument analysiert die **MUSS-Anforderungen** aus dem Projekt-Briefing ("Implementing Robust Non-negative Matrix Factorization (NMF) in Julia", 25. November 2025) und vergleicht sie mit dem aktuellen Implementierungsstand. Für alle noch offenen Anforderungen werden konkrete Implementierungspläne mit Code-Beispielen vorgeschlagen.

---

## Status-Übersicht: Anforderungen vs. Implementierung

### ✅ Vollständig erfüllt

| # | Anforderung | Status | Implementierung |
|---|-------------|--------|-----------------|
| 1.1 | Standard NMF mit multiplikativen Update-Regeln | ✅ | `src/StandardNMF.jl::nmf()` |
| 1.2 | Optionen: rank, maxiter, tol | ✅ | Alle Parameter vorhanden |
| 1.3 | Rückgabe: W, H, history | ✅ | Vollständig implementiert |
| 2.1 | Synthetische nicht-negative Daten | ✅ | `src/Data.jl::generate_synthetic_data()` |
| 2.2 | Daten mit Rauschen/Ausreißern | ✅ | `add_gaussian_noise!()`, `add_sparse_outliers!()` |
| 2.3 | Normalisierung für Non-Negativität | ✅ | `normalize_nonnegative!()` |
| 3.1 | Paketstruktur auf GitHub | ✅ | Vollständige Struktur vorhanden |
| 3.2 | Docstrings für alle Funktionen | ✅ | Alle exportierten Funktionen dokumentiert |
| 4.1 | Unit-Tests | ✅ | `test/test_data.jl` |
| 4.2 | Beispiel-Demo | ✅ | `examples/demo_robustnmf.jl` |

### ⚠️ Teilweise erfüllt

| # | Anforderung | Status | Fehlend |
|---|-------------|--------|---------|
| 1.4 | Robuste NMF (mind. EINE Variante: L1 ODER Huber ODER Itakura-Saito) | ✅ | IRLS-L1 implementiert ✓ |
| 3.3 | Visualisierung: Basis/Aktivierung | ⚠️ | Vorhanden, aber begrenzt |
| 3.4 | Visualisierung: Original/Noisy/Recon | ⚠️ | Vorhanden, aber begrenzt |
| 3.5 | Visualisierung: Error-Konvergenz | ⚠️ | Vorhanden, aber begrenzt |
| 5.1 | Tests: Non-Negativität | ⚠️ | Nur für Daten, nicht für NMF |
| 5.2 | Tests: Konvergenz | ⚠️ | Nicht explizit getestet |
| 5.3 | Tests: Reconstruction Accuracy | ⚠️ | Nicht explizit getestet |

### ❌ Nicht erfüllt

| # | Anforderung | Status | Grund |
|---|-------------|--------|-------|
| 2.4 | Laden realer Datasets (Gesichter) | ❌ | Funktion vorhanden, aber keine Datensätze |
| 4.1 | Performance-Evaluation (optional) | ❌ | Nicht implementiert |
| 4.2 | Metriken: RMSE, MAE, Explained Variance | ❌ | Nicht implementiert |
| 4.3 | Runtime-Vergleich | ❌ | Nicht implementiert |
| 6.1 | README.md mit Installation/Usage | ⚠️ | Vorhanden, aber minimal |
| 6.2 | GettingStarted.md | ⚠️ | In docs/src/index.md, nicht separat |
| 7.1 | Documenter.jl Dokumentation | ❌ | Struktur vorhanden, nicht gebaut |

---

## Detaillierte Anforderungsanalyse

### TASK 1: Data Preparation

#### ✅ Erfüllt
- Synthetische Daten: `generate_synthetic_data()`
- Gauß-Rauschen: `add_gaussian_noise!()`
- Spärliche Ausreißer: `add_sparse_outliers!()`
- Normalisierung: `normalize_nonnegative!()`

#### ❌ Fehlend: Reale Datensätze

**Anforderung (PDF Seite 2):**
> "Real datasets (e.g., grayscale images of faces)"

**Aktueller Stand:**
- `load_image_folder()` ist implementiert
- **ABER:** Keine Beispiel-Datensätze im Repository
- **ABER:** Keine Demonstration mit realen Daten

**Lösung:**
Entweder:
1. Einbindung eines öffentlichen Gesichter-Datensatzes (z.B. Olivetti Faces)
2. Download-Script für externe Datensätze
3. Beispiel-Bilder im Repository (mit Lizenz)

---

### TASK 2: Standard NMF Implementation

#### ✅ Vollständig erfüllt

Alle Anforderungen aus PDF Seite 2:
- ✅ Multiplikative Update-Regeln
- ✅ Minimierung von `‖X - WH‖²_F`
- ✅ Parameter: `rank`, `maxiter`, `tol`
- ✅ Rückgabe: `W, H, history`
- ✅ Rekonstruktion: `X_recon = W * H`

**Keine weiteren Maßnahmen erforderlich.**

---

### TASK 3: Robust NMF Implementation

#### ✅ ERFÜLLT - MUSS-Anforderung bereits implementiert!

**Anforderung (PDF Seite 2):**
> "Extend the algorithm to handle noise or outliers using **one of the following approaches**:
> - L1-norm objective: min_{W,H≥0} ‖X - WH‖₁  ✅ **IMPLEMENTIERT**
> - Huber loss: smooth transition between L1 and L2 norms
> - Itakura–Saito divergence"

**Aktueller Stand:**
- ✅ **IRLS-Approximation von L1-Norm implementiert** → **ANFORDERUNG ERFÜLLT**
- ⚠️ Huber-Loss nicht implementiert (OPTIONAL, nicht erforderlich)
- ⚠️ Itakura-Saito-Divergenz nicht implementiert (OPTIONAL, nicht erforderlich)

**Fazit:** Die MUSS-Anforderung "mindestens eine robuste Variante" ist durch `robust_nmf()` (L1/IRLS) **vollständig erfüllt**.

---

#### 🎯 OPTIONAL: Zusätzliche robuste Varianten

Die folgenden Implementierungen sind **NICE-TO-HAVE** und können das Projekt erweitern, sind aber **NICHT verpflichtend**:

#### 🎯 OPTIONAL-Feature 1: Huber-Loss NMF

**Mathematischer Hintergrund:**

Huber-Loss kombiniert L2 (für kleine Fehler) und L1 (für große Fehler):

```
ℓ_δ(r) = { ½r²           falls |r| ≤ δ
         { δ(|r| - ½δ)   falls |r| > δ
```

wobei δ der Schwellwert ist.

**Ziel-Funktion:**
```
min_{W,H≥0} Σᵢⱼ ℓ_δ(Xᵢⱼ - (WH)ᵢⱼ)
```

**Implementierungsplan:**

```julia
"""
    robust_nmf_huber(X; rank=2, maxiter=50, tol=1e-4, delta=1.0,
                     eps_update=1e-12, seed=0)

Robust NMF using Huber loss, which smoothly transitions between L2 (for
small residuals) and L1 (for large outliers).

# Arguments
- `X`: Non-negative data matrix (m × n)
- `rank`: Target rank (default: 2)
- `maxiter`: Maximum iterations (default: 50)
- `tol`: Convergence tolerance (default: 1e-4)
- `delta`: Huber threshold parameter (default: 1.0)
  - Smaller δ → more robust (more L1-like)
  - Larger δ → less robust (more L2-like)
- `eps_update`: Numerical stabilization (default: 1e-12)
- `seed`: Random seed (default: 0)

# Returns
- `W`: Basis matrix (m × rank)
- `H`: Coefficient matrix (rank × n)
- `history`: Huber loss per iteration

# Algorithm
Uses IRLS (Iteratively Reweighted Least Squares) with Huber weights:
    w(r) = 1                    if |r| ≤ δ
    w(r) = δ / |r|              if |r| > δ

# References
Huber, P. J. (1964). "Robust Estimation of a Location Parameter"
"""
function robust_nmf_huber(
    X::AbstractMatrix{<:Real};
    rank::Int = 2,
    maxiter::Int = 50,
    tol::Float64 = 1e-4,
    delta::Float64 = 1.0,
    eps_update::Float64 = 1e-12,
    seed::Int = 0
)
    @assert minimum(X) >= 0 "X must be non-negative"
    @assert delta > 0 "delta must be positive"

    Random.seed!(seed)

    m, n = size(X)
    W = rand(m, rank)
    H = rand(rank, n)

    history = zeros(Float64, maxiter)
    prev_loss = Inf
    eps_conv = eps(Float64)

    for iter in 1:maxiter
        # 1) Current reconstruction
        WH = W * H

        # 2) Residuals
        R = X .- WH

        # 3) Huber weights
        # w(r) = 1 if |r| ≤ δ, else w(r) = δ/|r|
        abs_R = abs.(R)
        V = ifelse.(abs_R .<= delta, 1.0, delta ./ (abs_R .+ eps_update))

        # 4) Weighted multiplicative updates
        numerH = W' * (V .* X)
        denomH = W' * (V .* WH) .+ eps_update
        H .*= numerH ./ denomH

        # 5) Update W
        WH = W * H  # recompute after H update
        R = X .- WH
        abs_R = abs.(R)
        V = ifelse.(abs_R .<= delta, 1.0, delta ./ (abs_R .+ eps_update))

        numerW = (V .* X) * H'
        denomW = (V .* WH) * H' .+ eps_update
        W .*= numerW ./ denomW

        # 6) Compute Huber loss
        WH = W * H
        R = X .- WH
        abs_R = abs.(R)
        huber_loss = sum(ifelse.(abs_R .<= delta,
                                 0.5 .* R.^2,
                                 delta .* (abs_R .- 0.5 * delta)))

        history[iter] = huber_loss

        # 7) Check convergence
        if abs(prev_loss - huber_loss) / (prev_loss + eps_conv) < tol
            history = history[1:iter]
            break
        end
        prev_loss = huber_loss
    end

    return W, H, history
end
```

**Integration in `src/RobustNMFAlgorithms.jl`:**
- Funktion hinzufügen
- In `src/RobustNMF.jl` exportieren
- Docstring vervollständigen

---

#### 🎯 OPTIONAL-Feature 2: Itakura-Saito Divergenz

**Mathematischer Hintergrund:**

Itakura-Saito (IS) Divergenz ist besonders geeignet für spektrale Daten (Audio, Signale):

```
D_IS(x|y) = x/y - log(x/y) - 1
```

**Ziel-Funktion:**
```
min_{W,H≥0} Σᵢⱼ D_IS(Xᵢⱼ | (WH)ᵢⱼ)
```

**Multiplikative Update-Regeln (aus Févotte et al. 2009):**

```
H ← H .* (W' * (X ./ (WH).²)) ./ (W' * (1 ./ WH))
W ← W .* ((X ./ (WH).²) * H') ./ ((1 ./ WH) * H')
```

**Implementierungsplan:**

```julia
"""
    robust_nmf_is(X; rank=2, maxiter=50, tol=1e-4, eps_update=1e-12, seed=0)

Robust NMF using Itakura-Saito (IS) divergence, particularly suited for
spectral data (audio, signals) where scale-invariance is important.

# Arguments
- `X`: Non-negative data matrix (m × n)
- `rank`: Target rank (default: 2)
- `maxiter`: Maximum iterations (default: 50)
- `tol`: Convergence tolerance (default: 1e-4)
- `eps_update`: Numerical stabilization (default: 1e-12)
- `seed`: Random seed (default: 0)

# Returns
- `W`: Basis matrix (m × rank)
- `H`: Coefficient matrix (rank × n)
- `history`: IS divergence per iteration

# Algorithm
Multiplicative updates derived from β-divergence with β=0:
    H ← H .* (W'×(X./(WH).²)) ./ (W'×(1./WH))
    W ← W .* ((X./(WH).²)×H') ./ ((1./WH)×H')

# References
Févotte, C., Bertin, N., & Durrieu, J.-L. (2009).
"Nonnegative matrix factorization with the Itakura-Saito divergence"
Neural Computation, 21(3), 793-830.

# Notes
- IS divergence is scale-invariant: D_IS(αx|αy) = D_IS(x|y)
- Particularly effective for audio and spectral data
- Requires X to have no zero entries (add small epsilon if needed)
"""
function robust_nmf_is(
    X::AbstractMatrix{<:Real};
    rank::Int = 2,
    maxiter::Int = 50,
    tol::Float64 = 1e-4,
    eps_update::Float64 = 1e-12,
    seed::Int = 0
)
    @assert minimum(X) >= 0 "X must be non-negative"

    # Ensure no zeros in X (IS divergence requires X > 0)
    if any(X .== 0)
        @warn "X contains zeros; adding eps_update to avoid division by zero"
        X = X .+ eps_update
    end

    Random.seed!(seed)

    m, n = size(X)
    W = rand(m, rank) .+ eps_update  # avoid zeros
    H = rand(rank, n) .+ eps_update

    history = zeros(Float64, maxiter)
    prev_div = Inf
    eps_conv = eps(Float64)

    for iter in 1:maxiter
        # 1) Current reconstruction
        WH = W * H .+ eps_update

        # 2) Update H
        # H ← H .* (W' * (X ./ WH²)) ./ (W' * (1 ./ WH))
        numerH = W' * (X ./ (WH .^ 2))
        denomH = W' * (1.0 ./ WH) .+ eps_update
        H .*= numerH ./ denomH
        H .= max.(H, eps_update)  # ensure positivity

        # 3) Update W
        WH = W * H .+ eps_update
        numerW = (X ./ (WH .^ 2)) * H'
        denomW = (1.0 ./ WH) * H' .+ eps_update
        W .*= numerW ./ denomW
        W .= max.(W, eps_update)

        # 4) Compute IS divergence
        WH = W * H .+ eps_update
        is_div = sum(X ./ WH .- log.(X ./ WH) .- 1.0)
        history[iter] = is_div

        # 5) Check convergence
        if abs(prev_div - is_div) / (abs(prev_div) + eps_conv) < tol
            history = history[1:iter]
            break
        end
        prev_div = is_div
    end

    return W, H, history
end
```

**Integration:**
- In `src/RobustNMFAlgorithms.jl` hinzufügen
- Export in `src/RobustNMF.jl`
- Demo-Beispiel für Audio/Spektraldaten

---

### TASK 4: Visualization

#### ⚠️ Teilweise erfüllt

**Anforderung (PDF Seite 2):**
> "Provide plotting utilities to visualize:
> - Basis vectors (W) and activation coefficients (H)
> - Original, noisy, and reconstructed data matrices
> - Error convergence over iterations"

**Aktueller Stand:**
- ✅ `plot_factors()`: W und H als Heatmaps
- ✅ `plot_reconstruction()`: X, X_noisy, W×H
- ✅ `plot_convergence()`: Error-Historie

#### 🎯 MUSS-Anforderung 1: Erweiterte Visualisierung

**Fehlende Features:**

1. **Basis-Vektoren als Bilder** (für Bild-NMF)
2. **Aktivierungs-Muster pro Sample**
3. **Vergleichs-Plots** mehrerer Algorithmen
4. **Residuen-Analyse**

**Implementierungsplan:**

```julia
# In src/Plotting.jl hinzufügen:

"""
    plot_basis_images(W, img_size; n_components=nothing, savepath=nothing,
                      display_plot=true)

Plot learned basis vectors as images (for image NMF).

# Arguments
- `W`: Basis matrix (n_pixels × rank)
- `img_size`: Tuple (height, width) of original image
- `n_components`: Number of components to plot (default: all, max 25)
- `savepath`: Optional save path
- `display_plot`: Display plot (default: true)

# Example
```julia
X, (h, w), _ = load_image_folder("faces/")
W, H, _ = nmf(X; rank=25)
plot_basis_images(W, (h, w); n_components=16)
```
"""
function plot_basis_images(
    W,
    img_size::Tuple{Int,Int};
    n_components=nothing,
    savepath=nothing,
    display_plot=true
)
    h, w = img_size
    n_total = size(W, 2)

    # Limit to max 25 components for readability
    if n_components === nothing
        n_components = min(n_total, 25)
    end
    n_components = min(n_components, n_total)

    # Determine grid layout
    n_cols = ceil(Int, sqrt(n_components))
    n_rows = ceil(Int, n_components / n_cols)

    plots_array = []
    for i in 1:n_components
        basis_vec = W[:, i]
        basis_img = reshape(basis_vec, h, w)

        p = heatmap(basis_img;
                   title="Component $i",
                   axis=false,
                   colorbar=false,
                   yflip=true,
                   aspect_ratio=:equal)
        push!(plots_array, p)
    end

    p = plot(plots_array...;
            layout=(n_rows, n_cols),
            size=(n_cols * 150, n_rows * 150))

    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end


"""
    plot_activation_patterns(H, sample_indices; savepath=nothing, display_plot=true)

Plot activation patterns for specific samples.

# Arguments
- `H`: Activation matrix (rank × n_samples)
- `sample_indices`: Vector of sample indices to plot
- `savepath`: Optional save path
- `display_plot`: Display plot (default: true)

# Example
```julia
W, H, _ = nmf(X; rank=10)
plot_activation_patterns(H, [1, 5, 10, 15])  # Plot activations for 4 samples
```
"""
function plot_activation_patterns(
    H,
    sample_indices;
    savepath=nothing,
    display_plot=true
)
    n_components = size(H, 1)
    plots_array = []

    for idx in sample_indices
        activations = H[:, idx]
        p = bar(1:n_components, activations;
               title="Sample $idx",
               xlabel="Component",
               ylabel="Activation",
               legend=false)
        push!(plots_array, p)
    end

    p = plot(plots_array...;
            layout=(1, length(sample_indices)),
            size=(length(sample_indices) * 250, 300))

    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end


"""
    plot_algorithm_comparison(results_dict; metric="error", savepath=nothing,
                              display_plot=true)

Compare convergence of multiple NMF algorithms.

# Arguments
- `results_dict`: Dict with algorithm names as keys and (W, H, history) as values
- `metric`: Metric name for y-axis label (default: "error")
- `savepath`: Optional save path
- `display_plot`: Display plot (default: true)

# Example
```julia
results = Dict(
    "Standard NMF" => nmf(X_noisy; rank=5),
    "Robust NMF (L1)" => robust_nmf(X_noisy; rank=5),
    "Robust NMF (Huber)" => robust_nmf_huber(X_noisy; rank=5)
)
plot_algorithm_comparison(results; metric="Reconstruction Error")
```
"""
function plot_algorithm_comparison(
    results_dict::Dict;
    metric="error",
    savepath=nothing,
    display_plot=true
)
    p = plot(; xlabel="Iteration", ylabel=metric, legend=:topright)

    for (name, (W, H, history)) in results_dict
        plot!(p, history; label=name, linewidth=2)
    end

    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end


"""
    plot_residuals_analysis(X, W, H; percentiles=[50, 75, 90, 95, 99],
                           savepath=nothing, display_plot=true)

Analyze reconstruction residuals with histogram and percentile markers.

# Arguments
- `X`: Original data
- `W`, `H`: Factor matrices
- `percentiles`: Percentiles to mark (default: [50, 75, 90, 95, 99])
- `savepath`: Optional save path
- `display_plot`: Display plot (default: true)

# Example
```julia
W, H, _ = nmf(X_noisy; rank=5)
plot_residuals_analysis(X, W, H)
```
"""
function plot_residuals_analysis(
    X,
    W,
    H;
    percentiles=[50, 75, 90, 95, 99],
    savepath=nothing,
    display_plot=true
)
    residuals = abs.(X .- W * H)
    flat_residuals = vec(residuals)

    # Histogram
    p1 = histogram(flat_residuals;
                  bins=50,
                  xlabel="Absolute Residual",
                  ylabel="Frequency",
                  title="Residuals Distribution",
                  legend=false,
                  fillalpha=0.6)

    # Add percentile markers
    for pct in percentiles
        val = quantile(flat_residuals, pct / 100)
        vline!(p1, [val];
              linestyle=:dash,
              linewidth=2,
              label="$pct%: $(round(val, digits=3))")
    end

    # Residuals heatmap
    p2 = heatmap(residuals;
                title="Residuals Heatmap",
                xlabel="Samples",
                ylabel="Features",
                colorbar=true)

    p = plot(p1, p2; layout=(1, 2), size=(1000, 400))
    return _finalize_plot(p; savepath=savepath, display_plot=display_plot)
end
```

**Export in `src/RobustNMF.jl`:**
```julia
export plot_basis_images,
       plot_activation_patterns,
       plot_algorithm_comparison,
       plot_residuals_analysis
```

---

### TASK 5: Performance Evaluation (Optional)

#### ❌ Nicht implementiert

**Anforderung (PDF Seite 3):**
> "Measure reconstruction error and robustness against:
> - Gaussian noise
> - Sparse outliers (random large entries)
>
> Quantify performance using RMSE, MAE, and explained variance.
> Compare runtime and convergence between standard and robust NMF."

#### 🎯 MUSS-Anforderung 2: Metriken-Modul

**Implementierungsplan:**

Neues Modul: `src/Metrics.jl`

```julia
using LinearAlgebra, Statistics

"""
    rmse(X, X_hat)

Root Mean Squared Error between X and X_hat.

RMSE = sqrt(mean((X - X_hat)²))
"""
function rmse(X::AbstractMatrix, X_hat::AbstractMatrix)
    @assert size(X) == size(X_hat) "Matrices must have same size"
    return sqrt(mean((X .- X_hat) .^ 2))
end


"""
    mae(X, X_hat)

Mean Absolute Error between X and X_hat.

MAE = mean(|X - X_hat|)
"""
function mae(X::AbstractMatrix, X_hat::AbstractMatrix)
    @assert size(X) == size(X_hat) "Matrices must have same size"
    return mean(abs.(X .- X_hat))
end


"""
    explained_variance(X, X_hat)

Explained variance ratio: proportion of variance in X explained by X_hat.

Explained Variance = 1 - Var(X - X_hat) / Var(X)

Returns a value in [0, 1] where 1 means perfect reconstruction.
"""
function explained_variance(X::AbstractMatrix, X_hat::AbstractMatrix)
    @assert size(X) == size(X_hat) "Matrices must have same size"

    residual_var = var(vec(X .- X_hat))
    total_var = var(vec(X))

    if total_var == 0
        return total_var == residual_var ? 1.0 : 0.0
    end

    return 1.0 - residual_var / total_var
end


"""
    relative_error(X, X_hat; norm_type=2)

Relative reconstruction error: ‖X - X_hat‖ / ‖X‖

# Arguments
- `X`: Original matrix
- `X_hat`: Reconstructed matrix
- `norm_type`: Norm type (default: 2 for Frobenius)

Returns a value in [0, ∞) where 0 means perfect reconstruction.
"""
function relative_error(X::AbstractMatrix, X_hat::AbstractMatrix; norm_type=2)
    @assert size(X) == size(X_hat) "Matrices must have same size"

    X_norm = norm(X, norm_type)

    if X_norm == 0
        @warn "X has zero norm"
        return X == X_hat ? 0.0 : Inf
    end

    return norm(X .- X_hat, norm_type) / X_norm
end


"""
    ReconstructionMetrics

Struct to hold all reconstruction metrics.
"""
struct ReconstructionMetrics
    rmse::Float64
    mae::Float64
    explained_variance::Float64
    relative_error_l2::Float64
    relative_error_l1::Float64
end


"""
    compute_metrics(X, X_hat)

Compute all reconstruction metrics at once.

# Returns
ReconstructionMetrics struct with fields:
- rmse
- mae
- explained_variance
- relative_error_l2
- relative_error_l1

# Example
```julia
metrics = compute_metrics(X_clean, W * H)
println("RMSE: ", metrics.rmse)
println("MAE: ", metrics.mae)
println("Explained Variance: ", metrics.explained_variance)
```
"""
function compute_metrics(X::AbstractMatrix, X_hat::AbstractMatrix)
    return ReconstructionMetrics(
        rmse(X, X_hat),
        mae(X, X_hat),
        explained_variance(X, X_hat),
        relative_error(X, X_hat; norm_type=2),
        relative_error(X, X_hat; norm_type=1)
    )
end


"""
    print_metrics(metrics::ReconstructionMetrics; name="")

Pretty-print reconstruction metrics.
"""
function print_metrics(metrics::ReconstructionMetrics; name="")
    title = isempty(name) ? "Reconstruction Metrics" : "Metrics: $name"
    println("\n", "="^50)
    println(title)
    println("="^50)
    println("  RMSE:                ", round(metrics.rmse, digits=6))
    println("  MAE:                 ", round(metrics.mae, digits=6))
    println("  Explained Variance:  ", round(metrics.explained_variance, digits=6))
    println("  Relative Error (L2): ", round(metrics.relative_error_l2, digits=6))
    println("  Relative Error (L1): ", round(metrics.relative_error_l1, digits=6))
    println("="^50, "\n")
end


# Export functions
export rmse, mae, explained_variance, relative_error,
       ReconstructionMetrics, compute_metrics, print_metrics
```

**Integration:**
```julia
# In src/RobustNMF.jl
include("Metrics.jl")

export rmse, mae, explained_variance, relative_error,
       ReconstructionMetrics, compute_metrics, print_metrics
```

---

#### 🎯 MUSS-Anforderung 3: Benchmark-Suite

Neues Modul: `src/Benchmarking.jl`

```julia
using BenchmarkTools, Statistics, Printf

"""
    benchmark_nmf_algorithms(X, algorithms_dict; rank=10, n_runs=3)

Benchmark multiple NMF algorithms on the same data.

# Arguments
- `X`: Data matrix
- `algorithms_dict`: Dict with algorithm names and functions
- `rank`: Factorization rank (default: 10)
- `n_runs`: Number of runs for timing (default: 3)

# Returns
DataFrame with columns: algorithm, time_mean, time_std, rmse, mae, ...

# Example
```julia
algorithms = Dict(
    "Standard" => (X; kw...) -> nmf(X; kw...),
    "Robust L1" => (X; kw...) -> robust_nmf(X; kw...),
    "Robust Huber" => (X; kw...) -> robust_nmf_huber(X; kw...)
)
results = benchmark_nmf_algorithms(X_noisy, algorithms; rank=5)
```
"""
function benchmark_nmf_algorithms(
    X::AbstractMatrix,
    algorithms_dict::Dict;
    rank::Int = 10,
    n_runs::Int = 3,
    maxiter::Int = 200
)
    X_clean = get(algorithms_dict, "clean_data", nothing)

    results = []

    println("\n", "="^70)
    println("Benchmarking NMF Algorithms")
    println("="^70)
    println("Data size: $(size(X))")
    println("Rank: $rank")
    println("Max iterations: $maxiter")
    println("Number of runs: $n_runs")
    println("="^70, "\n")

    for (name, algo_func) in algorithms_dict
        name == "clean_data" && continue  # skip metadata

        println("Running: $name...")

        # Timing
        times = Float64[]
        W, H, history = nothing, nothing, nothing

        for run in 1:n_runs
            t = @elapsed begin
                W, H, history = algo_func(X; rank=rank, maxiter=maxiter)
            end
            push!(times, t)
        end

        time_mean = mean(times)
        time_std = std(times)

        # Reconstruction metrics
        X_recon = W * H
        metrics = compute_metrics(X, X_recon)

        # Metrics against clean data (if provided)
        clean_metrics = nothing
        if X_clean !== nothing
            clean_metrics = compute_metrics(X_clean, X_recon)
        end

        # Store results
        result = Dict(
            "algorithm" => name,
            "time_mean" => time_mean,
            "time_std" => time_std,
            "n_iterations" => length(history),
            "final_loss" => history[end],
            "rmse" => metrics.rmse,
            "mae" => metrics.mae,
            "explained_var" => metrics.explained_variance,
            "rel_error_l2" => metrics.relative_error_l2
        )

        if clean_metrics !== nothing
            result["rmse_clean"] = clean_metrics.rmse
            result["mae_clean"] = clean_metrics.mae
            result["explained_var_clean"] = clean_metrics.explained_variance
        end

        push!(results, result)

        @printf("  Time: %.3f ± %.3f s\n", time_mean, time_std)
        @printf("  Iterations: %d\n", length(history))
        @printf("  RMSE: %.6f\n", metrics.rmse)
        @printf("  MAE: %.6f\n", metrics.mae)
        if clean_metrics !== nothing
            @printf("  RMSE (vs clean): %.6f\n", clean_metrics.rmse)
        end
        println()
    end

    println("="^70, "\n")

    return results
end


"""
    robustness_test(X_clean, noise_levels, outlier_fractions, algorithms_dict;
                   rank=10)

Test robustness of NMF algorithms under varying corruption levels.

# Arguments
- `X_clean`: Clean data matrix
- `noise_levels`: Vector of Gaussian noise sigma values
- `outlier_fractions`: Vector of sparse outlier fractions
- `algorithms_dict`: Dict with algorithm names and functions
- `rank`: Factorization rank (default: 10)

# Returns
Dict with noise/outlier levels as keys and benchmark results as values

# Example
```julia
results = robustness_test(
    X_clean,
    [0.0, 0.05, 0.1, 0.2],  # noise levels
    [0.0, 0.01, 0.05],      # outlier fractions
    algorithms
)
```
"""
function robustness_test(
    X_clean::AbstractMatrix,
    noise_levels::Vector{<:Real},
    outlier_fractions::Vector{<:Real},
    algorithms_dict::Dict;
    rank::Int = 10
)
    results = Dict()

    println("\n", "="^70)
    println("Robustness Test")
    println("="^70)

    for sigma in noise_levels
        for frac in outlier_fractions
            println("\nNoise σ = $sigma, Outlier fraction = $frac")
            println("-"^70)

            # Corrupt data
            X_corrupt = copy(X_clean)
            if sigma > 0
                add_gaussian_noise!(X_corrupt; sigma=sigma)
            end
            if frac > 0
                add_sparse_outliers!(X_corrupt; fraction=frac, magnitude=5.0)
            end

            # Add clean data reference
            algos_with_clean = merge(algorithms_dict, Dict("clean_data" => X_clean))

            # Benchmark
            bench_results = benchmark_nmf_algorithms(
                X_corrupt,
                algos_with_clean;
                rank=rank,
                n_runs=1
            )

            results[(sigma, frac)] = bench_results
        end
    end

    println("="^70, "\n")

    return results
end


export benchmark_nmf_algorithms, robustness_test
```

**Integration und Export:**
```julia
# In src/RobustNMF.jl
include("Benchmarking.jl")

export benchmark_nmf_algorithms, robustness_test
```

---

### TASK 6: Tests

#### ⚠️ Unvollständig

**Anforderung (PDF Seite 4):**
> "Unit tests in test/runtests.jl using Test.jl:
> - Validate non-negativity of factors
> - Convergence
> - Reconstruction accuracy
> - Test robustness on noisy data"

**Aktueller Stand:**
- ✅ Tests für Daten-Utilities
- ❌ **KEINE Tests für NMF-Algorithmen**

#### 🎯 MUSS-Anforderung 4: Umfassende Tests

Neue Datei: `test/test_nmf.jl`

```julia
using Test
using RobustNMF
using LinearAlgebra, Statistics

@testset "Standard NMF" begin

    @testset "Basic functionality" begin
        m, n, r = 20, 15, 5
        X, W_true, H_true = generate_synthetic_data(m, n; rank=r, seed=42)

        W, H, history = nmf(X; rank=r, maxiter=100, seed=42)

        # Test dimensions
        @test size(W) == (m, r)
        @test size(H) == (r, n)
        @test length(history) <= 100

        # Test non-negativity
        @test all(W .>= 0)
        @test all(H .>= 0)

        # Test reconstruction
        X_recon = W * H
        @test size(X_recon) == size(X)
    end

    @testset "Convergence" begin
        X, _, _ = generate_synthetic_data(30, 20; rank=5, seed=123)
        W, H, history = nmf(X; rank=5, maxiter=500, tol=1e-6)

        # Error should decrease monotonically (or stay same)
        for i in 2:length(history)
            @test history[i] <= history[i-1] + 1e-10  # small tolerance for numerical errors
        end

        # Final error should be small
        @test history[end] < history[1]
    end

    @testset "Reconstruction accuracy on clean data" begin
        m, n, r = 25, 20, 3
        X, W_true, H_true = generate_synthetic_data(m, n; rank=r, noise_level=0.0, seed=1)

        W, H, history = nmf(X; rank=r, maxiter=500, tol=1e-8)

        # Should achieve near-perfect reconstruction on clean, low-rank data
        rel_error = norm(X - W * H) / norm(X)
        @test rel_error < 1e-4
    end

    @testset "Invalid inputs" begin
        # Negative entries
        X_neg = [-1.0 2.0; 3.0 4.0]
        @test_throws AssertionError nmf(X_neg; rank=2)

        # Rank too large
        X_small = rand(5, 5)
        # Should not error, but won't be meaningful
        W, H, _ = nmf(X_small; rank=3, maxiter=10)
        @test size(W) == (5, 3)
    end

end


@testset "Robust NMF (L1/IRLS)" begin

    @testset "Basic functionality" begin
        m, n, r = 20, 15, 4
        X, _, _ = generate_synthetic_data(m, n; rank=r, seed=42)

        W, H, history = robust_nmf(X; rank=r, maxiter=100, seed=42)

        # Test dimensions
        @test size(W) == (m, r)
        @test size(H) == (r, n)

        # Test non-negativity
        @test all(W .>= 0)
        @test all(H .>= 0)
    end

    @testset "Robustness to outliers" begin
        m, n, r = 30, 25, 5
        X_clean, _, _ = generate_synthetic_data(m, n; rank=r, seed=1)

        # Add strong outliers
        X_corrupt = copy(X_clean)
        add_sparse_outliers!(X_corrupt; fraction=0.05, magnitude=10.0, seed=1)

        # Compare standard vs robust
        W_std, H_std, _ = nmf(X_corrupt; rank=r, maxiter=200)
        W_rob, H_rob, _ = robust_nmf(X_corrupt; rank=r, maxiter=200, seed=1)

        # Robust should have lower MAE on clean data
        mae_std = mean(abs.(X_clean - W_std * H_std))
        mae_rob = mean(abs.(X_clean - W_rob * H_rob))

        @test mae_rob < mae_std  # Robust should be better
    end

    @testset "Convergence" begin
        X, _, _ = generate_synthetic_data(25, 20; rank=4, seed=456)
        W, H, history = robust_nmf(X; rank=4, maxiter=300, tol=1e-6)

        # MAE should decrease
        @test history[end] < history[1]
    end

end


@testset "Metrics" begin

    X = rand(10, 10)
    X_hat = X .+ 0.1 .* randn(10, 10)

    @testset "RMSE" begin
        err = rmse(X, X_hat)
        @test err > 0
        @test rmse(X, X) == 0.0
    end

    @testset "MAE" begin
        err = mae(X, X_hat)
        @test err > 0
        @test mae(X, X) == 0.0
    end

    @testset "Explained Variance" begin
        ev = explained_variance(X, X_hat)
        @test 0 <= ev <= 1
        @test explained_variance(X, X) == 1.0
    end

    @testset "Relative Error" begin
        rel_err = relative_error(X, X_hat)
        @test rel_err > 0
        @test relative_error(X, X) == 0.0
    end

    @testset "compute_metrics" begin
        metrics = compute_metrics(X, X_hat)
        @test metrics.rmse > 0
        @test metrics.mae > 0
        @test 0 <= metrics.explained_variance <= 1
    end

end


@testset "Integration Tests" begin

    @testset "Full workflow: synthetic data" begin
        # Generate
        X, W_true, H_true = generate_synthetic_data(40, 30; rank=6, seed=99)

        # Corrupt
        add_gaussian_noise!(X; sigma=0.05)

        # Factorize
        W, H, history = nmf(X; rank=6, maxiter=300)

        # Metrics
        metrics = compute_metrics(X, W * H)

        @test metrics.rmse > 0
        @test metrics.explained_variance > 0.8  # Should explain most variance
    end

end
```

**Integration in `test/runtests.jl`:**
```julia
using Test
using RobustNMF

include("test_data.jl")
include("test_nmf.jl")  # NEU
```

---

### TASK 7: Dokumentation

#### ⚠️ Unvollständig

**Anforderung (PDF Seite 4):**
> "- Docstrings for all exported functions ✅
> - A README.md and GettingStarted.md ⚠️
> - Optional: generate docs via Documenter.jl ❌"

#### 🎯 MUSS-Anforderung 5: README.md erweitern

**Aktueller Stand:**
Minimales README vorhanden, aber:
- Keine Installation-Anleitung
- Keine Beispiele
- Keine Feature-Übersicht
- Keine Badges/CI-Status

**Verbesserter README.md:**

```markdown
# RobustNMF.jl

[![Build Status](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/stable/)
[![Coverage](https://codecov.io/gh/TU-Julia-RobustNMF/RobustNMF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TU-Julia-RobustNMF/RobustNMF.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Non-negative Matrix Factorization (NMF)** with robust variants for noisy and corrupted data.

## Features

- ⚡ **Standard NMF**: Fast multiplicative updates (Frobenius norm)
- 🛡️ **Robust NMF**: Multiple loss functions for outlier resistance
  - L1-norm (IRLS)
  - Huber loss
  - Itakura-Saito divergence
- 📊 **Comprehensive Tools**:
  - Synthetic data generation
  - Noise injection (Gaussian, sparse outliers)
  - Image loading and preprocessing
  - Visualization utilities
  - Performance metrics (RMSE, MAE, explained variance)
- 🧪 **Well-tested**: Extensive unit tests and benchmarks
- 📚 **Documented**: Full API documentation with examples

## Installation

### From GitHub (Development)

```julia
using Pkg
Pkg.add(url="https://github.com/TU-Julia-RobustNMF/RobustNMF.jl")
```

### Local Development

```bash
git clone https://github.com/TU-Julia-RobustNMF/RobustNMF.jl.git
cd RobustNMF.jl
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using RobustNMF
```

## Quick Start

### Basic Example

```julia
using RobustNMF

# Generate synthetic data
X, W_true, H_true = generate_synthetic_data(100, 80; rank=10, seed=42)

# Standard NMF
W, H, history = nmf(X; rank=10, maxiter=500)

# Reconstruction
X_recon = W * H

# Metrics
using LinearAlgebra
error = norm(X - X_recon)
println("Reconstruction error: $error")
```

### Robust NMF with Noisy Data

```julia
using RobustNMF, Statistics

# Clean data
X_clean, _, _ = generate_synthetic_data(50, 40; rank=5, seed=1)

# Add corruption
X_noisy = copy(X_clean)
add_gaussian_noise!(X_noisy; sigma=0.1)
add_sparse_outliers!(X_noisy; fraction=0.02, magnitude=10.0)

# Compare algorithms
W_std, H_std, hist_std = nmf(X_noisy; rank=5, maxiter=200)
W_rob, H_rob, hist_rob = robust_nmf(X_noisy; rank=5, maxiter=200)

# Evaluate against clean data
println("Standard NMF MAE: ", mean(abs.(X_clean - W_std * H_std)))
println("Robust NMF MAE:   ", mean(abs.(X_clean - W_rob * H_rob)))

# Visualize
using Plots
plot_convergence([hist_std, hist_rob]; labels=["Standard", "Robust"])
```

### Image Factorization

```julia
using RobustNMF

# Load face images
X, (h, w), filenames = load_image_folder("path/to/faces/"; pattern=".png")

# Extract basis faces
W, H, history = nmf(X; rank=25, maxiter=1000)

# Visualize basis images
plot_basis_images(W, (h, w); n_components=16, savepath="basis_faces.png")
```

## Documentation

Full documentation available at: [https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/](https://TU-Julia-RobustNMF.github.io/RobustNMF.jl/)

- [Getting Started Guide](docs/src/index.md)
- [API Reference](docs/src/functions.md)
- [Examples](examples/)

## Algorithms

### Standard NMF (Frobenius Norm)

Minimizes: `‖X - WH‖²_F`

Update rules:
```
H ← H .* (W'X) ./ (W'WH + ε)
W ← W .* (XH') ./ (WHH' + ε)
```

### Robust NMF (L1-Norm / IRLS)

Minimizes: `‖X - WH‖₁`

Uses iteratively reweighted least squares with adaptive weights for outlier downweighting.

### Robust NMF (Huber Loss)

Smooth transition between L2 (small errors) and L1 (large errors):
```
ℓ_δ(r) = ½r²           if |r| ≤ δ
ℓ_δ(r) = δ(|r| - ½δ)   if |r| > δ
```

### Robust NMF (Itakura-Saito Divergence)

Specialized for audio/spectral data. Scale-invariant divergence.

## Performance

Typical convergence on 100×100 matrix with rank 10:
- **Standard NMF**: 50-200 iterations, ~0.1-0.5 seconds
- **Robust NMF**: 100-300 iterations, ~0.2-1.0 seconds

## Project Structure

```
RobustNMF.jl/
├── src/
│   ├── RobustNMF.jl              # Main module
│   ├── StandardNMF.jl            # Standard NMF
│   ├── RobustNMFAlgorithms.jl    # Robust variants
│   ├── Data.jl                   # Data utilities
│   ├── Plotting.jl               # Visualization
│   ├── Metrics.jl                # Performance metrics
│   └── Benchmarking.jl           # Benchmarking tools
├── test/                         # Unit tests
├── examples/                     # Example scripts
├── docs/                         # Documentation
└── Project.toml
```

## Testing

```julia
using Pkg
Pkg.test("RobustNMF")
```

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{robustnmf_jl,
  author = {Samaan, Haitham},
  title = {RobustNMF.jl: Robust Non-negative Matrix Factorization in Julia},
  year = {2025},
  url = {https://github.com/TU-Julia-RobustNMF/RobustNMF.jl}
}
```

## References

- Lee & Seung (2001). "Algorithms for non-negative matrix factorization"
- Hamza & Brady (2006). "Reconstruction of reflectance spectra using robust nonnegative matrix factorization"
- Févotte et al. (2009). "Nonnegative matrix factorization with the Itakura-Saito divergence"
- Zhang et al. (2011). "Robust non-negative matrix factorization"

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Author**: Haitham Samaan (h.samaan@campus.tu-berlin.de)
**Course**: Julia Programming for Machine Learning, TU Berlin
**Project**: Final Project, Winter Semester 2025/26
```

---

#### 🎯 MUSS-Anforderung 6: GettingStarted.md

Neue Datei: `GettingStarted.md`

```markdown
# Getting Started with RobustNMF.jl

This guide will help you get up and running with RobustNMF.jl.

## Installation

### Prerequisites

- Julia ≥ 1.11
- Git (for cloning repository)

### Install from GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/TU-Julia-RobustNMF/RobustNMF.jl")
```

### Local Development

```bash
git clone https://github.com/TU-Julia-RobustNMF/RobustNMF.jl.git
cd RobustNMF.jl
julia
```

```julia
] activate .
] instantiate
using RobustNMF
```

## Your First NMF

### Step 1: Generate Data

```julia
using RobustNMF

# Create 50×40 matrix with rank-5 structure
X, W_true, H_true = generate_synthetic_data(50, 40; rank=5, seed=42)

println("Data size: ", size(X))
println("True rank: ", size(W_true, 2))
```

### Step 2: Factorize

```julia
# Run NMF with rank 5
W, H, history = nmf(X; rank=5, maxiter=300, tol=1e-5)

println("Learned W size: ", size(W))
println("Learned H size: ", size(H))
println("Converged in ", length(history), " iterations")
```

### Step 3: Evaluate

```julia
using LinearAlgebra

# Reconstruction
X_recon = W * H

# Error
error = norm(X - X_recon)
println("Reconstruction error: ", round(error, digits=4))

# Relative error
rel_error = norm(X - X_recon) / norm(X)
println("Relative error: ", round(rel_error, digits=6))
```

### Step 4: Visualize

```julia
using Plots

# Plot convergence
plot_convergence(history; label="Standard NMF", savepath="convergence.png")

# Plot factors
plot_factors(W, H; savepath="factors.png")

# Plot reconstruction
plot_reconstruction(X, X, W, H; savepath="reconstruction.png")
```

## Working with Noisy Data

### Add Noise

```julia
# Clean data
X_clean, _, _ = generate_synthetic_data(60, 50; rank=5, seed=1)

# Copy and corrupt
X_noisy = copy(X_clean)
add_gaussian_noise!(X_noisy; sigma=0.1)           # Gaussian noise
add_sparse_outliers!(X_noisy; fraction=0.02, magnitude=10.0)  # Outliers
```

### Compare Algorithms

```julia
# Standard NMF (L2 loss)
W_std, H_std, hist_std = nmf(X_noisy; rank=5, maxiter=200)

# Robust NMF (L1 loss)
W_rob, H_rob, hist_rob = robust_nmf(X_noisy; rank=5, maxiter=200)

# Huber loss (if implemented)
W_hub, H_hub, hist_hub = robust_nmf_huber(X_noisy; rank=5, maxiter=200, delta=0.5)
```

### Evaluate Robustness

```julia
using Statistics

# Metrics against clean data
mae_std = mean(abs.(X_clean - W_std * H_std))
mae_rob = mean(abs.(X_clean - W_rob * H_rob))
mae_hub = mean(abs.(X_clean - W_hub * H_hub))

println("Standard NMF MAE: ", round(mae_std, digits=6))
println("Robust NMF MAE:   ", round(mae_rob, digits=6))
println("Huber NMF MAE:    ", round(mae_hub, digits=6))

# Visualization
plot_convergence([hist_std, hist_rob, hist_hub];
                labels=["Standard", "Robust L1", "Huber"],
                savepath="comparison.png")
```

## Working with Images

### Load Images

```julia
# Load grayscale face images
X, (h, w), filenames = load_image_folder("path/to/faces/";
                                         pattern=".png",
                                         normalize=true)

println("Loaded $(length(filenames)) images")
println("Image size: $(h)×$(w)")
println("Data matrix: $(size(X))")  # (h*w × n_images)
```

### Extract Basis Images

```julia
# Factorize: extract 25 "eigenfaces"
W, H, history = nmf(X; rank=25, maxiter=1000, tol=1e-6)

# Visualize basis images
plot_basis_images(W, (h, w); n_components=16, savepath="basis_faces.png")
```

### Reconstruct Images

```julia
# Reconstruct first image
img_idx = 1
img_original = reshape(X[:, img_idx], h, w)
img_recon = reshape((W * H)[:, img_idx], h, w)

# Plot comparison
using Plots
p1 = heatmap(img_original; title="Original", yflip=true)
p2 = heatmap(img_recon; title="Reconstructed", yflip=true)
plot(p1, p2; layout=(1,2), size=(800, 400))
savefig("image_reconstruction.png")
```

## Performance Metrics

### Compute All Metrics

```julia
# Factorize
W, H, _ = nmf(X_noisy; rank=5, maxiter=200)

# Compute metrics
metrics = compute_metrics(X_clean, W * H)

# Print
print_metrics(metrics; name="Standard NMF")

# Access individual metrics
println("RMSE: ", metrics.rmse)
println("MAE: ", metrics.mae)
println("Explained Variance: ", metrics.explained_variance)
```

### Benchmark Multiple Algorithms

```julia
using RobustNMF

# Define algorithms
algorithms = Dict(
    "Standard NMF" => (X; kw...) -> nmf(X; kw...),
    "Robust NMF (L1)" => (X; kw...) -> robust_nmf(X; kw...),
    "Robust NMF (Huber)" => (X; kw...) -> robust_nmf_huber(X; kw...)
)

# Run benchmark
results = benchmark_nmf_algorithms(X_noisy, algorithms; rank=5, n_runs=3)

# Results is a vector of dicts with timing and metric information
```

## Advanced Topics

### Custom Initialization

```julia
# Try multiple random seeds and pick best
using LinearAlgebra

seeds = [1, 42, 123, 456, 789]
best_error = Inf
best_W, best_H = nothing, nothing

for seed in seeds
    W, H, _ = nmf(X; rank=5, maxiter=200, seed=seed)
    error = norm(X - W * H)

    if error < best_error
        best_error = error
        best_W, best_H = W, H
    end
end

println("Best reconstruction error: ", best_error)
```

### Hyperparameter Tuning

```julia
# Test different ranks
ranks = [3, 5, 7, 10, 15]
errors = Float64[]

for r in ranks
    W, H, _ = nmf(X; rank=r, maxiter=300)
    push!(errors, norm(X - W * H))
end

# Plot elbow curve
using Plots
plot(ranks, errors;
    marker=:circle,
    xlabel="Rank",
    ylabel="Reconstruction Error",
    title="Rank Selection (Elbow Method)",
    legend=false)
savefig("rank_selection.png")
```

## Troubleshooting

### Issue: Slow Convergence

**Solution**: Increase `maxiter` or relax `tol`

```julia
W, H, history = nmf(X; rank=10, maxiter=1000, tol=1e-3)
```

### Issue: Poor Reconstruction

**Possible causes**:
1. Rank too low → Increase rank
2. Data not suitable for NMF → Check non-negativity
3. Need more iterations → Increase `maxiter`

### Issue: Negative Values in Factors

This should never happen. If it does:
- Check that input `X` is non-negative
- Report as bug

### Issue: "X must be non-negative" Error

**Solution**: Normalize your data first

```julia
normalize_nonnegative!(X; rescale=true)
W, H, _ = nmf(X; rank=5)
```

## Next Steps

- Read the [full documentation](docs/)
- Explore [examples/](examples/)
- Check out the [API reference](docs/src/functions.md)
- Run the demo: `julia examples/demo_robustnmf.jl`

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl/issues)
- **Email**: h.samaan@campus.tu-berlin.de

Happy factorizing! 🎉
```

---

## Zusammenfassung: Prioritäten

### ✅ BEREITS ERFÜLLT

- ✅ **Robuste NMF** (L1/IRLS) - Mindestanforderung erfüllt!

### 🔴 KRITISCH (MUSS für finales Projekt)

1. **Erweiterte Visualisierung** (`src/Plotting.jl`, Seite 14-16)
2. **Metriken-Modul** (`src/Metrics.jl`, Seite 16-17)
3. **Benchmark-Suite** (`src/Benchmarking.jl`, Seite 17-19)
4. **NMF-Tests** (`test/test_nmf.jl`, Seite 20-22)
5. **README.md erweitern** (Seite 23-24)
6. **GettingStarted.md erstellen** (Seite 25-27)

### 🟡 WICHTIG (Sollte vorhanden sein)

7. **Reale Datensätze** einbinden oder Download-Script
8. **Documenter.jl Build** konfigurieren
9. Demo-Script erweitern mit neuen Features

### 🟢 OPTIONAL (Nice-to-have, NICHT verpflichtend)

10. **Huber-Loss NMF** (zusätzliche robuste Variante)
11. **Itakura-Saito NMF** (zusätzliche robuste Variante für Audio)
12. Performance-Optimierungen (In-place, GPU)
13. Zusätzliche Initialisierungsmethoden (NNDSVD)
14. Regularisierung (L1/L2 auf W, H)
15. CI/CD Pipeline erweitern

---

## Implementierungs-Roadmap

**WICHTIG:** Die robuste NMF-Anforderung ist bereits erfüllt! Die folgenden Phasen konzentrieren sich auf die verbleibenden MUSS-Anforderungen.

### Phase 1: Metriken und Benchmarking (1 Tag)
- [ ] `src/Metrics.jl` erstellen
- [ ] `src/Benchmarking.jl` erstellen
- [ ] Integration und Export
- [ ] Basis-Tests für Metriken

### Phase 2: Tests erweitern (1 Tag)
- [ ] `test/test_nmf.jl` erstellen
- [ ] Standard-NMF testen (Non-Negativität, Konvergenz, Accuracy)
- [ ] Robust-NMF testen (Robustheit gegen Ausreißer)
- [ ] Metriken-Tests
- [ ] Integration-Tests

### Phase 3: Visualisierung erweitern (1 Tag)
- [ ] `plot_basis_images()` implementieren
- [ ] `plot_activation_patterns()` implementieren
- [ ] `plot_algorithm_comparison()` implementieren
- [ ] `plot_residuals_analysis()` implementieren
- [ ] Integration und Export

### Phase 4: Dokumentation (1 Tag)
- [ ] README.md überarbeiten (Features, Beispiele, Installation)
- [ ] GettingStarted.md erstellen (Tutorial)
- [ ] Docstrings vervollständigen
- [ ] Documenter.jl Build testen (optional)

### Phase 5: Integration und Demo (1 Tag)
- [ ] `examples/demo_robustnmf.jl` erweitern
- [ ] Neue Visualisierungen integrieren
- [ ] Metriken und Benchmarks demonstrieren
- [ ] Alle Plots generieren
- [ ] Finale Tests durchführen

### Phase 6 (OPTIONAL): Zusätzliche Features
- [ ] Huber-Loss NMF implementieren (wenn gewünscht)
- [ ] Itakura-Saito NMF implementieren (wenn gewünscht)
- [ ] Reale Datensätze einbinden
- [ ] Performance-Optimierungen

---

**Geschätzter Gesamtaufwand (MUSS-Anforderungen)**: 5 Arbeitstage

**Geschätzter Gesamtaufwand (mit OPTIONAL)**: 7-8 Arbeitstage

**Nächste Schritte**: Mit Phase 1 (Metriken) beginnen, da diese für Tests und Benchmarks benötigt werden.
