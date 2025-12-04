using RobustNMF
using Test

@testset "RobustNMF.jl" begin
    @testset "timestwo tests" begin
        @test timestwo(2) == 4
    end
end
