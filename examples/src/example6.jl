using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA, BenchmarkTools
CUDA.allowscalar(false)

a = Float32(42)
X = Float32[1,2]
Y = Float32[2,3]

using LinearAlgebra

LinearAlgebra.axpy!(a,X,copy(Y))


dX = CuArray(X)
dY = CuArray(Y)

CUBLAS.axpy!(length(dY),a,dX,copy(dY))


# directly calling C level code that is wrapped in julia
handle = CUBLAS.handle()
CUBLAS.cublasSaxpy_v2(handle, length(dY), Ref(a), dX, stride(X,1), dY, stride(dY,1))

dY



devices()

dev = CuDevice(0)

capability(dev)

totalmem(dev) |> Base.format_bytes

attribute(dev, CUDA.DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)

