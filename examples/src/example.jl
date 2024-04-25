# %% [markdown]
# # Reference
# This notebook follows CSCS [Course](https://github.com/omlins/julia-gpu-course-2023/tree/main)
# # Why
# Julia is a high level language which is very easy to write. And, GPU
# programming provides speedup to your code without designing new algorithm.
#
#


# %%
using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA
CUDA.versioninfo()

# Specify version of cuda, incase you wanna use a more advanced version
# CUDA.set_runtime_version!(version; local_toolkit=true)
#
# Asks julia to use local tool-kit instead of one on the hpc
# JULIA_DEBUG=CUDA_Runtime_Discovery \ julia -e 'using CUDA'


function axpy!(z, a, x, y)
    z .= a .* x .+ y
end

x = [1, 2]
y = [2, 3]

alpha = 4

z = similar(x)

axpy!(z, alpha, x, y)

x = CuArray([1, 2])
y = CuArray([2, 3])
alpha = 4
z = similar(x)

axpy!(z, alpha, x, y)

display(z)

function axpy_kernel!(z, a, x, y)
    function kernel(z, a, x, y)
        i = threadIdx().x
        if i <= length(z)
            @inbounds z[i] = a * x[i] + y[i]
        end
        return
    end
    @cuda threads=length(z) kernel(z, a, x, y)

    return z
end


x = CuArray([1, 2])
y = CuArray([2, 3])
alpha = 4
z = similar(x)

axpy_kernel!(z, alpha, x, y)


function axpy_kernel!(z, a, x, y)
    function kernel(z, a, x, y)
        i = threadIdx().x
        if i <= length(z)
            @inbounds z[i] = a * x[i] + y[i]
        end
        return
    end
    kernel = @cuda launch=false kernel(z,a,x,y)
    threads = min(length(z),256)
    blocks = cld(length(z), threads)
    kernel(z,a,x,y; threads, blocks)

    return z
end

x = CuArray([1, 2])
y = CuArray([2, 3])
alpha = 4
z = similar(x)

axpy_kernel!(z, alpha, x, y)


x = CUDA.rand(Float32, 4096, 4096)
y = CUDA.rand(Float32, 4096, 4096)
z = similar(x)
alpha = rand(Float32)
axpy_kernel!(z, alpha, x,y)
CUDA.@profile trace=true axpy_kernel!(z, alpha, x, y)


axpy!(z,alpha,x,y)
CUDA.@profile trace=true axpy!(z,alpha,x,y)


x = CuArray([1,2])
y = CuArray([2,3])
alpha = 4

# cuBLAS do not support F64
CUBLAS.axpy!(length(y),alpha, x, y)

x = CuArray([1f0, 2f0])
y = CuArray([2f0, 3f0])
alpha = 4
CUBLAS.axpy!(length(y),alpha, x, y)

# not robust benchmarking since you only run once
CUDA.@profile trace=true CUBLAS.axpy!(length(y),alpha, x, y)

