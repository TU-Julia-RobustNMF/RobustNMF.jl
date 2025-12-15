using RobustNMF
using Documenter

DocMeta.setdocmeta!(RobustNMF, :DocTestSetup, :(using RobustNMF); recursive=true)

makedocs(;
    modules=[RobustNMF],
    authors="Haitham Samaan <h.samaan@campus.tu-berlin.de>, Adrian Brag",
    sitename="RobustNMF.jl",
    format=Documenter.HTML(;
        canonical="https://tu-julia-robustnmf.github.io/RobustNMF.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Getting Started" => "index.md",
        "Functions" => "functions.md",
    ],
)

deploydocs(;
    deploy_config = Documenter.GitHubActions(),
)
