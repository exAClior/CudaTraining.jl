using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA, BenchmarkTools
CUDA.allowscalar(false)

# kernel analysis and optimization

# good pattern for packaging kernel function with host function
# because other users' won't be able to call the kernel function directly
# and use a non-optimal configuration
function rmse(A::AbstractArray{T}, B::AbstractArray{T}) where T
    @assert size(A) == size(B)
    C = similar(A,1)
    C .= 0

    function rmse_kernel(C,A,B)
        i = (blockIdx().x -1) * blockDim().x + threadIdx().x
        stride = gridDim().x + blockDim().x

        while i <= length(A)
            a = A[i]
            b = B[i]
            CUDA.@atomic C[] += (a-b)^2
            i += stride
        end
    end

    let kernel = @cuda launch=false rmse_kernel(C,A,B)
        config = CUDA.launch_configuration(kernel.fun)
        threads = min(length(A),config.threads)
        blocks = min(cld(length(A),threads),config.blocks)
        kernel(C,A,B; threads,blocks)    
    end
    return CUDA.@allowscalar sqrt(C[] / length(A))
end

A = CUDA.rand(2048, 2048)
B = CUDA.rand(2048, 2048)

@benchmark rmse(A,B)

CUDA.@profile rmse(A,B)

# kernel usage is high but performance is not good
# meaning the algorithm is bad
# You should use NSight Compute to check.