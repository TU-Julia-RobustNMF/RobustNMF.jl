using RobustNMF
using Documenter

DocMeta.setdocmeta!(RobustNMF, :DocTestSetup, :(using RobustNMF); recursive=true)

makedocs(;
    modules=[RobustNMF],
    authors="Haitham Samaan <h.samaan@campus.tu-berlin.de>, Adrian Brag",
    sitename="RobustNMF.jl",
    format=Documenter.HTML(;
        canonical="https://hai-sam.github.io/RobustNMF.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hai-sam/RobustNMF.jl",
    devbranch="master",
)
