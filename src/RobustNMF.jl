module RobustNMF

include("DataPreparation.jl")
using .DataPreparation: generate_synthetic_data, add_noise_and_outliers, normalize_data

export generate_synthetic_data, add_noise_and_outliers, normalize_data

end # module RobustNMF
