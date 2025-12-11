module RobustNMF

include("DataPreparation.jl")
using .DataPreparation
export generate_synthetic_data, add_noise_and_outliers, normalize_data

include("StandardNMF.jl")
using .StandardNMF
export nmf

end # module RobustNMF
