# RobustNMF.jl - Vollständige Projektdokumentation

## Projektübersicht

**RobustNMF.jl** ist ein Julia-Paket zur Implementierung von **Non-negative Matrix Factorization (NMF)** mit besonderem Fokus auf Robustheit gegenüber Rauschen und Ausreißern. Das Projekt wurde im Rahmen des Kurses "Julia Programming for Machine Learning" an der TU Berlin entwickelt.

### Projektkontext

Dieses Projekt implementiert die Anforderungen aus dem Dokument "Implementing Robust Non-negative Matrix Factorization (NMF) in Julia" (25. November 2025). Es handelt sich um ein vollständig funktionales Julia-Paket mit standardisierter Struktur, Tests, Dokumentation und Beispielen.

---

## Mathematischer Hintergrund

### Non-negative Matrix Factorization (NMF)

NMF ist eine Dimensionsreduktionstechnik, die eine nicht-negative Datenmatrix **X ∈ ℝ^(m×n)** als Produkt zweier niedrig-rangiger nicht-negativer Matrizen approximiert:

```
X ≈ W × H
```

Dabei gilt:
- **W ∈ ℝ^(m×r)**: Basismatrix (Basis-Vektoren/Features)
- **H ∈ ℝ^(r×n)**: Aktivierungsmatrix (Koeffizienten/Gewichte)
- **r**: Rang der Faktorisierung (typischerweise r ≪ min(m,n))

### Anwendungsbereiche

- **Bildverarbeitung**: Extraktion von Bildbausteinen und Gesichtsmerkmalen
- **Topic Modeling**: Dokumentenanalyse und Themenextraktion
- **Bioinformatik**: Genexpressionsdaten und biologische Signale
- **Audioverarbeitung**: Quelltrennung und Spektralanalyse

Die Nicht-Negativitätsbedingung führt oft zu interpretierbaren, teilebasierten Repräsentationen.

---

## Projektstruktur

```
RobustNMF.jl/
├── Project.toml              # Paket-Metadaten und Abhängigkeiten
├── Manifest.toml             # Versionsspezifische Abhängigkeiten
├── README.md                 # Projekt-Readme
├── NEXT_STEPS_PLAN.md        # Entwicklungsplan und offene Punkte
├── PROJEKTBESCHREIBUNG.md    # Diese Datei
│
├── src/                      # Quellcode
│   ├── RobustNMF.jl         # Hauptmodul (exportiert alle Funktionen)
│   ├── StandardNMF.jl       # Standard-NMF Implementierung
│   ├── RobustNMFAlgorithms.jl  # Robuste NMF-Varianten
│   ├── Data.jl              # Daten-Utilities (Generierung, Laden, Preprocessing)
│   └── Plotting.jl          # Visualisierungsfunktionen
│
├── test/                     # Unit-Tests
│   ├── runtests.jl          # Test-Runner
│   └── test_data.jl         # Tests für Daten-Utilities
│
├── examples/                 # Beispielskripte
│   └── demo_robustnmf.jl    # Vollständige Demo
│
└── docs/                     # Dokumentation (Documenter.jl)
    ├── make.jl              # Dokumentationsgenerator
    ├── Project.toml
    └── src/
        ├── index.md         # Dokumentations-Hauptseite
        └── functions.md     # API-Referenz
```

---

## Implementierte Features

### 1. Standard-NMF (`StandardNMF.jl`)

#### Funktion: `nmf()`

**Signatur:**
```julia
nmf(X; rank=10, maxiter=500, tol=1e-4, seed=0, eps_update=1e-12) -> (W, H, history)
```

**Beschreibung:**
Implementiert die Standard-NMF mit multiplikativen Update-Regeln zur Minimierung der Frobenius-Norm (L2-Verlust):

```
min_{W,H≥0} ‖X - W×H‖²_F
```

**Update-Regeln:**
```julia
H ← H .* (W'×X) ./ (W'×W×H + ε)
W ← W .* (X×H') ./ (W×H×H' + ε)
```

**Parameter:**
- `X`: Nicht-negative Eingabematrix (m×n)
- `rank`: Ziel-Rang der Faktorisierung (default: 10)
- `maxiter`: Maximale Anzahl an Iterationen (default: 500)
- `tol`: Konvergenztoleranz für relative Fehleränderung (default: 1e-4)
- `seed`: Random Seed für Reproduzierbarkeit (default: 0)
- `eps_update`: Numerischer Stabilisierungsterm (default: 1e-12)

**Rückgabewerte:**
- `W`: Basismatrix (m × rank)
- `H`: Aktivierungsmatrix (rank × n)
- `history`: Vektor mit Frobenius-Fehler pro Iteration

**Konvergenzkriterium:**
Der Algorithmus stoppt, wenn die relative Änderung des Rekonstruktionsfehlers unter `tol` fällt:
```julia
|error_prev - error_current| / error_prev < tol
```

**Eigenschaften:**
- Garantiert nicht-negative Faktoren (W ≥ 0, H ≥ 0)
- Monoton fallender Rekonstruktionsfehler
- Schnelle Konvergenz bei sauberen Daten
- **Sensibel gegenüber Ausreißern und starkem Rauschen** (L2-Verlust quadriert Fehler)

---

### 2. Robuste NMF (`RobustNMFAlgorithms.jl`)

#### Funktion: `robust_nmf()`

**Signatur:**
```julia
robust_nmf(X; rank=2, maxiter=50, tol=1e-4, eps_weight=1e-6, eps_update=1e-12, seed=0)
    -> (W, H, history)
```

**Beschreibung:**
Implementiert eine robuste NMF-Variante mittels IRLS (Iteratively Reweighted Least Squares) zur Approximation eines L1-Verlust-Ziels:

```
min_{W,H≥0} ‖X - W×H‖₁
```

Der L1-Verlust ist weniger empfindlich gegenüber Ausreißern als der L2-Verlust, da große Fehler nur linear (statt quadratisch) gewichtet werden.

**Algorithmus (IRLS):**

1. **Rekonstruktion**: Berechne `WH = W × H`
2. **Residuen**: Berechne `R = X - WH`
3. **Adaptive Gewichte**: Berechne `V = 1 / max(|R|, ε_weight)`
   - Kleine Residuen → große Gewichte
   - Große Residuen (Ausreißer) → kleine Gewichte
4. **Gewichtete Update-Regeln**:
   ```julia
   H ← H .* (W'×(V.*X)) ./ (W'×(V.*(W×H)) + ε)
   W ← W .* ((V.*X)×H') ./ ((V.*(W×H))×H' + ε)
   ```
5. **Konvergenz**: Überwache Mean Absolute Error (MAE)

**Parameter:**
- `X`: Nicht-negative Eingabematrix
- `rank`: Ziel-Rang (default: 2)
- `maxiter`: Maximale Iterationen (default: 50)
- `tol`: Konvergenztoleranz (default: 1e-4)
- `eps_weight`: Minimales Gewicht zur Stabilisierung (default: 1e-6)
- `eps_update`: Numerischer Stabilisierungsterm (default: 1e-12)
- `seed`: Random Seed (default: 0)

**Rückgabewerte:**
- `W`, `H`: Faktormatrizen
- `history`: Vektor mit Mean Absolute Error (L1) pro Iteration

**Vorteile:**
- **Robustheit**: Weniger empfindlich gegenüber Ausreißern und starkem Rauschen
- **Adaptive Gewichtung**: Problematische Einträge werden automatisch heruntergewichtet
- **Interpretierbarkeit**: Ähnlich zu Standard-NMF, aber stabiler

**Nachteile:**
- Etwas langsamere Konvergenz als Standard-NMF
- Zusätzlicher Hyperparameter `eps_weight`

---

### 3. Daten-Utilities (`Data.jl`)

#### 3.1 Synthetische Datengenerierung

**Funktion:** `generate_synthetic_data()`

```julia
generate_synthetic_data(m, n; rank=10, noise_level=0.0, seed=nothing)
    -> (X, W, H)
```

**Beschreibung:**
Generiert synthetische nicht-negative Datenmatrizen durch Sampling von zufälligen Faktoren W und H aus Uniform(0,1) und berechnet X = W×H.

**Parameter:**
- `m`: Anzahl Zeilen
- `n`: Anzahl Spalten
- `rank`: Rang der Faktorisierung (default: 10)
- `noise_level`: Standardabweichung von Gauß-Rauschen (default: 0.0)
- `seed`: Optional, Random Seed

**Rückgabe:**
- `X`: Generierte Datenmatrix (m × n)
- `W`: Ground-Truth Basismatrix (m × rank)
- `H`: Ground-Truth Aktivierungsmatrix (rank × n)

**Anwendung:**
Ideal zum Testen und Validieren von NMF-Algorithmen, da die wahren Faktoren bekannt sind.

---

#### 3.2 Rauschgenerierung

**Funktion:** `add_gaussian_noise!()`

```julia
add_gaussian_noise!(X; sigma=0.1, clip_at_zero=true) -> X
```

**Beschreibung:**
Fügt der Matrix X Gauß-Rauschen mit Standardabweichung `sigma` hinzu (in-place Operation).

**Parameter:**
- `X`: Zu störende Matrix (wird modifiziert!)
- `sigma`: Standardabweichung des Rauschens (default: 0.1)
- `clip_at_zero`: Negative Werte auf 0 clippen (default: true)

**Algorithmus:**
```julia
noise ~ N(0, σ²)
X ← X + noise
X ← max(X, 0)  # wenn clip_at_zero=true
```

---

**Funktion:** `add_sparse_outliers!()`

```julia
add_sparse_outliers!(X; fraction=0.01, magnitude=5.0, seed=nothing) -> X
```

**Beschreibung:**
Fügt spärliche, große positive Ausreißer zu einem Bruchteil der Einträge hinzu.

**Parameter:**
- `X`: Zu störende Matrix (wird modifiziert!)
- `fraction`: Anteil der zu modifizierenden Einträge (default: 0.01 = 1%)
- `magnitude`: Maximale Amplitude der Ausreißer (default: 5.0)
- `seed`: Optional, Random Seed

**Algorithmus:**
```julia
k = round(fraction × m × n)  # Anzahl Ausreißer
indices = sample(1:m×n, k)   # Zufällige Positionen
X[indices] += Uniform(0, magnitude)
```

**Anwendung:**
Simuliert korrupte Daten mit einzelnen stark fehlerhaften Messwerten, ideal zur Evaluierung der Robustheit von NMF-Algorithmen.

---

#### 3.3 Normalisierung

**Funktion:** `normalize_nonnegative!()`

```julia
normalize_nonnegative!(X; rescale=true) -> X
```

**Beschreibung:**
Transformiert Matrix X in-place, sodass alle Werte nicht-negativ sind.

**Algorithmus:**
1. **Shift**: Wenn min(X) < 0, dann X ← X - min(X)
2. **Rescale** (optional): X ← X / max(X), sodass Werte in [0, 1] liegen

**Parameter:**
- `X`: Zu normalisierende Matrix (wird modifiziert!)
- `rescale`: Auf [0,1] skalieren (default: true)

**Anwendung:**
Vorbereitung von Daten für NMF-Algorithmen, die nicht-negative Eingaben erfordern.

---

#### 3.4 Bildladen

**Funktion:** `load_image_folder()`

```julia
load_image_folder(dir; pattern="*.png", normalize=true)
    -> (X, (height, width), filenames)
```

**Beschreibung:**
Lädt alle Bilder aus einem Verzeichnis, konvertiert sie zu Graustufen, flacht sie ab und stapelt sie als Spalten einer Datenmatrix.

**Parameter:**
- `dir`: Pfad zum Bildverzeichnis
- `pattern`: Dateiendung (default: "*.png")
- `normalize`: Normalisierung auf [0,1] (default: true)

**Rückgabe:**
- `X`: Datenmatrix (height×width × num_images)
  - Jede Spalte ist ein abgeflachtes Graustufenbild
- `(height, width)`: Originale Bildgröße
- `filenames`: Liste der Dateinamen

**Anforderungen:**
- Alle Bilder müssen dieselbe Größe haben
- Unterstützt: PNG, JPG, etc. (via FileIO/ImageIO)

**Anwendung:**
Laden von Bilddatensätzen (z.B. Gesichter) zur Anwendung von NMF auf realen Daten.

---

### 4. Visualisierung (`Plotting.jl`)

#### 4.1 Faktor-Plots

**Funktion:** `plot_factors()`

```julia
plot_factors(W, H; clims_W=nothing, clims_H=nothing, savepath=nothing,
             display_plot=true, heatmap_kwargs...) -> plot
```

**Beschreibung:**
Visualisiert die Faktormatrizen W und H als nebeneinander liegende Heatmaps.

**Parameter:**
- `W`, `H`: Zu plottende Faktormatrizen
- `clims_W`, `clims_H`: Farbskalenlimits (optional, für Vergleiche)
- `savepath`: Optional, Speicherpfad für die Grafik
- `display_plot`: Plot anzeigen (default: true)
- `heatmap_kwargs...`: Zusätzliche Argumente für Heatmap

**Ausgabe:**
- Nebeneinander: Heatmap von W (links) und H (rechts)
- W-Achsen: x=components, y=features
- H-Achsen: x=samples, y=components

**Anwendung:**
Visualisierung der gelernten Basis-Vektoren und Aktivierungsmuster.

---

#### 4.2 Rekonstruktions-Plots

**Funktion:** `plot_reconstruction()`

```julia
plot_reconstruction(X, X_noisy, W, H; savepath=nothing, display_plot=true,
                    shared_clims=true, heatmap_kwargs...) -> plot
```

**Beschreibung:**
Vergleicht die originalen Daten X, die verrauschten Daten X_noisy und die Rekonstruktion W×H nebeneinander.

**Parameter:**
- `X`: Originale Daten
- `X_noisy`: Verrauschte Daten
- `W`, `H`: Faktormatrizen
- `shared_clims`: Einheitliche Farbskala für alle drei Plots (default: true)
- `savepath`: Speicherpfad (optional)
- `display_plot`: Plot anzeigen (default: true)

**Ausgabe:**
Drei nebeneinander liegende Heatmaps:
1. **X**: Originale Daten
2. **X_noisy**: Korrupte Daten
3. **W×H**: NMF-Rekonstruktion

**Anwendung:**
Visueller Vergleich der Rekonstruktionsqualität und Rauschunterdrückung.

---

#### 4.3 Konvergenz-Plots

**Funktion:** `plot_convergence()`

```julia
# Einzelne Kurve
plot_convergence(history; label="error", savepath=nothing, display_plot=true) -> plot

# Mehrere Kurven
plot_convergence(histories; labels=nothing, savepath=nothing, display_plot=true) -> plot
```

**Beschreibung:**
Visualisiert die Fehlerhistorie über die Iterationen.

**Parameter:**
- `history` / `histories`: Vektor(en) mit Fehlerwerten pro Iteration
- `label` / `labels`: Beschriftung(en) für die Kurve(n)
- `savepath`, `display_plot`: Wie oben

**Ausgabe:**
Linienplot mit:
- x-Achse: Iteration
- y-Achse: Fehler (Frobenius-Norm oder MAE)

**Anwendung:**
- Monitoring der Konvergenzgeschwindigkeit
- Vergleich verschiedener Algorithmen oder Hyperparameter

---

### 5. Beispiel-Demo (`examples/demo_robustnmf.jl`)

Das Demo-Skript demonstriert den vollständigen Workflow:

#### Workflow:

1. **Datengenerierung:**
   ```julia
   X, W_true, H_true = generate_synthetic_data(60, 50; rank=5, seed=1)
   ```

2. **Datenkorruption:**
   ```julia
   X_noisy = copy(X)
   add_gaussian_noise!(X_noisy; sigma=0.02)
   add_sparse_outliers!(X_noisy; fraction=0.01, magnitude=8.0)
   ```

3. **Faktorisierung:**
   ```julia
   W_std, H_std, hist_std = nmf(X_noisy; rank=5, maxiter=200)
   W_rob, H_rob, hist_rob = robust_nmf(X_noisy; rank=5, maxiter=200)
   ```

4. **Evaluierung:**
   ```julia
   # Frobenius-Norm (L2)
   err_std = norm(X - W_std * H_std)
   err_rob = norm(X - W_rob * H_rob)

   # Mean Absolute Error (L1)
   l1_std = mean(abs.(X - W_std * H_std))
   l1_rob = mean(abs.(X - W_rob * H_rob))
   ```

5. **Visualisierung:**
   - Faktormatrizen (mit und ohne gemeinsame Farbskala)
   - Robustheitsvergleich (Original, Verrauscht, Rekonstruktion, Residuen)
   - Konvergenzkurven

**Output:**
Das Demo erzeugt mehrere Plots im `plots/` Verzeichnis:
- `factors_standard.png` / `factors_robust.png`
- `factors_standard_shared.png` / `factors_robust_shared.png`
- `robustness_comparison.png` (8-Panel-Vergleich)
- `convergence_standard.png` / `convergence_robust.png`

---

## Tests (`test/`)

### Test-Suite (`test_data.jl`)

Die Tests validieren alle Daten-Utilities:

#### 1. `generate_synthetic_data`
- Korrekte Dimensionen von W, H, X
- Nicht-Negativität von X
- Exakte Reproduktion bei noise_level=0

#### 2. `add_gaussian_noise!`
- Matrix wird modifiziert
- Nicht-Negativität nach Clipping

#### 3. `normalize_nonnegative!`
- Minimum ≥ 0 nach Normalisierung
- Maximum ≈ 1 mit rescale=true

#### 4. `load_image_folder`
- Korrekte Dimensionen (Pixel × Anzahl Bilder)
- Korrekte Anzahl geladener Dateien

**Ausführung:**
```julia
] test
```

---

## Technische Details

### Abhängigkeiten (Project.toml)

**Core-Pakete:**
- `LinearAlgebra`: Matrix-Operationen
- `Random`: Zufallszahlengenerierung
- `Statistics`: Statistische Funktionen

**Datenverarbeitung:**
- `Images`: Bildmanipulation
- `FileIO`, `ImageIO`: Bildladen
- `ColorTypes`: Farbkonvertierung

**Visualisierung:**
- `Plots`: Plotting-Backend

**Versionen:**
- Julia: ≥ 1.11
- Alle Pakete mit kompatiblen Versionen gepinnt

---

### Paket-Installation

#### Aus lokalem Repository:

```julia
using Pkg
Pkg.activate("path/to/RobustNMF.jl")
Pkg.instantiate()
using RobustNMF
```

#### Für Entwicklung:

```julia
] dev path/to/RobustNMF.jl
using RobustNMF
```

---

## Verwendungsbeispiele

### Beispiel 1: Einfache NMF auf synthetischen Daten

```julia
using RobustNMF

# Synthetische Daten generieren
X, W_true, H_true = generate_synthetic_data(100, 80; rank=10, seed=42)

# Standard-NMF anwenden
W, H, history = nmf(X; rank=10, maxiter=500, tol=1e-4)

# Rekonstruktion
X_recon = W * H

# Rekonstruktionsfehler
error = norm(X - X_recon)
println("Reconstruction error: $error")

# Konvergenz visualisieren
plot_convergence(history; label="Standard NMF")
```

---

### Beispiel 2: Robustheitsevaluierung

```julia
using RobustNMF, LinearAlgebra, Statistics

# Saubere Daten
X_clean, _, _ = generate_synthetic_data(50, 40; rank=5, seed=1)

# Kopie erstellen und Rauschen hinzufügen
X_noisy = copy(X_clean)
add_gaussian_noise!(X_noisy; sigma=0.05)
add_sparse_outliers!(X_noisy; fraction=0.02, magnitude=10.0)

# Beide Algorithmen vergleichen
W_std, H_std, hist_std = nmf(X_noisy; rank=5, maxiter=200)
W_rob, H_rob, hist_rob = robust_nmf(X_noisy; rank=5, maxiter=200)

# Evaluierung gegen saubere Daten
println("Standard NMF:")
println("  L2 error: ", norm(X_clean - W_std * H_std))
println("  L1 error: ", mean(abs.(X_clean - W_std * H_std)))

println("\nRobust NMF:")
println("  L2 error: ", norm(X_clean - W_rob * H_rob))
println("  L1 error: ", mean(abs.(X_clean - W_rob * H_rob)))

# Visualisierung
plot_reconstruction(X_clean, X_noisy, W_std, H_std;
                    savepath="comparison_standard.png")
plot_reconstruction(X_clean, X_noisy, W_rob, H_rob;
                    savepath="comparison_robust.png")
```

---

### Beispiel 3: Bilddaten laden und faktorisieren

```julia
using RobustNMF

# Bilder laden (z.B. Gesichter)
X, (h, w), filenames = load_image_folder("path/to/faces/";
                                         pattern=".png",
                                         normalize=true)

println("Loaded $(length(filenames)) images of size $(h)×$(w)")
println("Data matrix size: $(size(X))")

# NMF anwenden (extrahiere z.B. 25 "Eigenfaces")
W, H, history = nmf(X; rank=25, maxiter=1000)

# W enthält jetzt 25 Basis-Gesichter
# Jedes kann als Bild rekonstruiert werden:
using Plots
basis_face_1 = reshape(W[:, 1], h, w)
heatmap(basis_face_1; title="Basis Face 1", yflip=true)
```

---

### Beispiel 4: Hyperparameter-Tuning

```julia
using RobustNMF

X, _, _ = generate_synthetic_data(50, 40; rank=5, noise_level=0.1, seed=1)

# Teste verschiedene Ränge
ranks = [3, 5, 7, 10]
errors = Float64[]

for r in ranks
    W, H, _ = nmf(X; rank=r, maxiter=300)
    err = norm(X - W * H)
    push!(errors, err)
    println("Rank $r: error = $err")
end

# Bester Rang (Elbow-Methode)
using Plots
plot(ranks, errors; marker=:circle, label="Reconstruction Error",
     xlabel="Rank", ylabel="Frobenius Norm")
```

---

## Performance-Charakteristiken

### Laufzeitkomplexität

#### Standard-NMF:
- Pro Iteration: **O(mnr + mr² + nr²)**
  - `W'×X`: O(mnr)
  - `W'×W×H`: O(mr² + nr²)
  - `X×H'`: O(mnr)
  - `W×H×H'`: O(mr² + nr²)

#### Robust-NMF:
- Pro Iteration: **O(mnr + mr² + nr²)** (gleiche Asymptotik)
- Zusätzlicher Overhead für Gewichtsberechnung: O(mn)

### Speicherkomplexität

- **Eingabe X**: O(mn)
- **Faktoren W, H**: O(mr + nr)
- **Temporäre Arrays**: O(mn) für Rekonstruktion und Residuen

### Typische Konvergenz

| Algorithmus   | Iterationen (saubere Daten) | Iterationen (verrauschte Daten) |
|---------------|------------------------------|----------------------------------|
| Standard-NMF  | 50-200                       | 100-500                          |
| Robust-NMF    | 100-300                      | 150-500                          |

---

## Bekannte Einschränkungen und offene Punkte

### Aktuelle Limitationen

1. **Algorithmus-Varianten:**
   - Nur eine robuste Variante implementiert (IRLS-L1)
   - Keine Implementierung von Huber-Loss oder Itakura-Saito-Divergenz

2. **Bildlader:**
   - Pattern-Matching mit `endswith()` kann problematisch sein
   - Keine explizite Prüfung auf Ordnerexistenz
   - Alle Bilder müssen identische Größe haben

3. **Initialisierung:**
   - Nur zufällige Initialisierung (Uniform)
   - Keine NNDSVD oder andere fortgeschrittene Initialisierungsmethoden

4. **Regularisierung:**
   - Keine L1/L2-Regularisierung auf W oder H
   - Keine Sparsity-Constraints

5. **Normalisierung:**
   - Keine optionale Normalisierung von W oder H während der Iteration
   - Skalierungsambiguität zwischen W und H

### Geplante Verbesserungen (siehe NEXT_STEPS_PLAN.md)

1. **API-Vereinheitlichung:**
   - Konsistente Keyword-Argumente
   - Entfernung nicht-implementierter Exports

2. **Algorithmus-Erweiterungen:**
   - Huber-Loss-Variante
   - Beta-Divergenzen (inkl. Itakura-Saito)
   - Regularisierungsoptionen

3. **Robustheit:**
   - Verbesserte Fehlerbehandlung
   - Mehr Validierung von Eingabedaten
   - Warnungen bei schlechter Konvergenz

4. **Performance:**
   - Optionale GPU-Unterstützung (via CUDA.jl)
   - In-place Updates zur Speicherreduktion
   - Parallele Initialisierung mit mehreren Seeds

5. **Evaluierung:**
   - Benchmark-Suite gegen andere NMF-Implementierungen
   - Mehr quantitative Metriken (z.B. Amari-Distance für Faktorvergleich)

---

## Wissenschaftliche Grundlagen

Das Projekt basiert auf folgenden wissenschaftlichen Arbeiten:

### Grundlegende NMF-Theorie:
- **Lee & Seung (2001)**: "Algorithms for non-negative matrix factorization"
  - Einführung der multiplikativen Update-Regeln
  - Beweis der Konvergenz und Monotonie

### Robuste NMF-Varianten:
- **Hamza & Brady (2006)**: "Reconstruction of reflectance spectra using robust nonnegative matrix factorization"
  - Anwendung robuster Loss-Funktionen auf NMF

- **Zhang et al. (2011)**: "Robust non-negative matrix factorization"
  - L1-Norm und andere robuste Zielfunktionen

- **Févotte et al. (2009)**: "Nonnegative matrix factorization with the Itakura-Saito divergence"
  - Spezialisierte Divergenzen für Audio-Anwendungen

- **Gao et al. (2015)**: "Robust capped norm nonnegative matrix factorization"
  - Capped-Norm-Ansätze für extreme Ausreißer

---

## Fazit

**RobustNMF.jl** ist ein vollständig funktionales, gut strukturiertes Julia-Paket zur Non-negative Matrix Factorization mit Fokus auf Robustheit. Es erfüllt alle Anforderungen des Kursprojekts:

✅ Standard-NMF mit multiplikativen Update-Regeln
✅ Robuste NMF-Variante (IRLS-L1)
✅ Synthetische Datengenerierung und Rauschmodelle
✅ Bildladen für reale Datensätze
✅ Umfassende Visualisierungsfunktionen
✅ Unit-Tests und Dokumentation
✅ Beispiel-Demo mit quantitativer Evaluierung
✅ Installierbare Paketstruktur auf GitHub

Das Paket ist bereit für weitere Entwicklung und kann als Basis für fortgeschrittene NMF-Forschung und -Anwendungen dienen.

---

## Kontakt und Mitwirkung

**Autor:** Haitham Samaan (h.samaan@campus.tu-berlin.de)
**Kurs:** Julia Programming for Machine Learning, TU Berlin
**Projekt-Repository:** [GitHub - RobustNMF.jl](https://github.com/TU-Julia-RobustNMF/RobustNMF.jl)

Contributions und Issues sind willkommen!

---

**Letzte Aktualisierung:** Januar 2026
**Julia-Version:** 1.11+
**Lizenz:** MIT (siehe Repository)
