# %% [markdown]
# # kernel programming
# You write a function that produces a scalar value. Instead of running a loop over the elements of the array, the kernel function is ran on threads on a warp... 

using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA, BenchmarkTools
CUDA.allowscalar(false)

# define a kernel function, plain julia function
function my_kernel()
    return
end

# run this kernel function
@cuda my_kernel()

# but the inputs of the kernel function needs to be bitstype and not a pointer to memory
isbitstype("hello")

# unless it's CuArray, which we know what and how to handle the actual data in the memory
# see Adapt.jl for conversion
A = CuArray([1,2,3])
isbits(A)

function my_kernel(A)
    return 
end
@cuda my_kernel(A)

function my_kernel(A)
    A[threadIdx().x] = threadIdx().x
    return
end

A = CUDA.zeros(10)

@cuda threads=length(A) my_kernel(A)

A

# how many threads does  your device support
attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

function kernel(A)
    i = (blockIdx().x -1) * blockDim().x + threadIdx().x
    if i <= length(A)
        A[i] = (; thread=threadIdx().x, block=blockIdx().x)
    end
    return
end

A = CuArray{@NamedTuple{thread::Int, block::Int}}(undef, 10)
@cuda threads=5 blocks=2 kernel(A)

A

# compile a kernel that is callable 
k = @cuda launch=false kernel(A)
@show config = launch_configuration(k.fun)
threads = min(length(A), config.threads)
blocks = cld(length(A), threads)
k(A; threads,blocks)
A

# synchronization is important in gpu programming
function reverse_kernel(A)
    @cushow i = threadIdx().x
    if i <= length(A)
        j = length(A) - i + 1
        A[i] = A[j]
    end
    return
end

A = CuArray(1:100)
@cuda threads=length(A) reverse_kernel(A)

Array(A) == 100:-1:1

# shared memory
function reverse_kernel(A::AbstractVector{T}) where {T}
    i = threadIdx().x
    j = length(A) - i + 1
    # use shared memory within a block
    # buf = CuStaticSharedArray(T,100)
    buf = CuDynamicSharedArray(T,length(A))
    buf[i] = A[i]
    sync_threads()
    A[i] = buf[j]
    return
end

A = CuArray(1:100)
@cuda threads =length(A) shmem=sizeof(A) reverse_kernel(A)
Array(A) == 100:-1:1

# Atomic operation to allow GPU to use global memory , very expensive
A_sum = CUDA.zeros(1)
A = CUDA.rand(512)

function kernel(A, A_sum)
    i = threadIdx().x
    CUDA.@atomic A_sum[] += A[i]
    return
end

@cuda threads=length(A) kernel(A, A_sum)
Array(A_sum)[]

# structured ouput

function kernel()
    i = threadIdx().x
    @cuprintf "I'm thread %ld\n" Int(i)
    return
end

function kernel()
    i = @cushow threadIdx().x
    return
end

@cuda kernel()

# return things, you cannot do this in cuda programming, however you may write the output to a length 1 array

function kernel(A)
    A[1] = 42
    return
end

A = CuArray([0])
@cuda kernel(A)
A

function kernel(ref)
    ref[] = 42
    return
end

# or you could use a ref here
ref = Ref(0)
@cuda kernel(ref)
ref

# there might be runtime error due to type mismatch
function kernel(a)
    if threadIdx().x == 1
        a[] += 42.1
    end
    return
end

# the location of error will not be displayed correctly and promptly
# because things execute on the GPU *Asynchroniously*
# this makes debugging difficult

@cuda kernel(CuArray[42])

# as a solution: julia -g2 will show stack trace

# unsupported IR, i.e the function to be called does not exist
function kernel(a)
    # typo should be threadIdx, hence the error 
    if threadId().x == 1
        a[] += 42.1
    end
    return
end
@cuda kernel(CuArray([1]))

@device_code_warntype @cuda kernel(CuArray([1]))

rmse(A,B) = sqrt(sum((A.-B).^2) / length(A))

A = rand(512)
B = rand(512)

rmse(A,B)

dA = CuArray(A)
dB = CuArray(B)
dC = similar(dA,1)

function rmse_kernel(C,A,B)
    i = threadIdx().x

    if i == 1
        C[] = 0
    end
    sync_threads()

    a= A[i]
    b= B[i]
    CUDA.@atomic C[] += (a-b)^2
    sync_threads()

    if i == 1
        C[1] = sqrt(C[] /length(A))
    end
    return

end


@cuda threads=length(dA) rmse_kernel(dC,dA,dB)
CUDA.@allowscalar dC[]


function rmse_kernel(C,A,B)
    i = (threadIdx().x-1) * blockDim().x + threadIdx().x

    if i == 1
        C[] = 0
    end
    sync_threads()

    a= A[i]
    b= B[i]
    CUDA.@atomic C[] += (a-b)^2
    sync_threads()

    if i == 1
        C[1] = sqrt(C[] /length(A))
    end
    return

end

k = @cuda launch=false rmse_kernel(dC,dA,dB)
config = CUDA.launch_configuration(k.fun)
threads = min(length(A),config.threads)
blocks = cld(length(A),threads)
k(dC, dA,dB; threads,blocks)

CUDA.@allowscalar dC[]

function rmse_kernel(C,A,B)
    i = (threadIdx().x-1) * blockDim().x + threadIdx().x
    
    if i > length(A)
        return 
    end

    a= A[i]
    b= B[i]
    CUDA.@atomic C[] += (a-b)^2

    return
end


function rmse_kernel(C,A,B)
    i = (threadIdx().x-1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x    

    while i<= length(input)
        i += stride
    end

    return
end

using KernelAbstractions

@kernel function ka_rmse_kernel(C,A,B)
    i = @index(Global)

    a = A[i]
    b = B[i]

    KernelAbstractions.@atomic C[]+= (a-b)^2
end


A = rand(512,512)
B = rand(512,512)

@show rmse(A,B)

# supports more than just NVIDIA GPUs, but also CPUs 
# but is in general slower
backend = CPU()
dA = KernelAbstractions.allocate(backend, eltype(A), size(A))
KernelAbstractions.copyto!(backend, dA, A)
dB = KernelAbstractions.allocate(backend, eltype(A), size(A))
KernelAbstractions.copyto!(backend, dB, B)

dC = KernelAbstractions.zeros(backend, eltype(A),1)

@show typeof(dC)

k = ka_rmse_kernel(backend)
k(dC,dA,dB; ndrange=size(A))

sqrt(Array(dC)[]/length(A))