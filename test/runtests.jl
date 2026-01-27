using Test
using RobustNMF
using Statistics  # Add this import for mean()

@testset verbose = true "RobustNMF.jl Test Suite" begin
    
    # Data utilities tests
    @testset "Data Utilities" begin
        include("test_data.jl")
    end
    
    # Standard NMF algorithm tests
    @testset "Standard NMF" begin
        include("test_standard_nmf.jl")
    end
   
    # L2,1-NMF algorithm tests
    @testset "Robust NMF (L2,1)" begin
        include("test_robust_nmf.jl")
    end
    
    # Visualization tests
    @testset "Plotting" begin
        include("test_plotting.jl")
    end
    
end
