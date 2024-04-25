using CudaTraining
using Documenter

DocMeta.setdocmeta!(CudaTraining, :DocTestSetup, :(using CudaTraining); recursive=true)

makedocs(;
    modules=[CudaTraining],
    authors="Yusheng Zhao <yushengzhao2020@outlook.com> and contributors",
    sitename="CudaTraining.jl",
    format=Documenter.HTML(;
        canonical="https://exAClior.github.io/CudaTraining.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/exAClior/CudaTraining.jl",
    devbranch="main",
)
