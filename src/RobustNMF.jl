module RobustNMF

include("Data.jl")

using .Data

export
generate_synthetic_data, 
add_gaussian_noise!, 
add_sparse_outliers!, 
normalize_nonnegative!, 
load_image_folder

include("StandardNMF.jl")
using .StandardNMF
export nmf, X_reconstruct

end # module RobustNMF
