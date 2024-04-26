using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA, BenchmarkTools
CUDA.allowscalar(false)