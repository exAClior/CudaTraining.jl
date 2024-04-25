using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA


#!!! it's a CPU object with pointer to GPU memory
A = CuArray([1. 2.; 3. 4.])

# retains the information about what transformation is applied to cu
# but didn't apply it. i.e the ' here
cu([1. 2.; 3. 4.]')

# didn't convert to Float64 and matrix is already transposed
CuArray([1. 2.; 3. 4.]') 

# using CUBLAS
A*A

# using native broadcast kernel

A .* A

a = rand(3)
b = rand(3)

broadcast(+, a,b)
a.+b # syntatic sugar
Meta.@lower f.(A .+ B) .- 2

map(A) do a
    a * 2
end

A = cu([1 2])

reduce(+,A)

A = cu([1 2; 3 4])
reduce(+, A; dims = 1)

A = CUDA.rand(10,10,3)

reduce(+ , A ; dims = [1,3])


A = cu(ones(5))
accumulate(+,A)

# since no parameter depends on CUDA array, need to let dispatcher know 
# that the output is a CUDA array by doing CUDA.zeros instead of zeros
CUDA.zeros(1)

CUDA.rand(Float64, 2,2)

A = CuArray(1:10)

A_sum = zero(eltype(A))

CUDA.allowscalar(true) # allow accessing array elements one by one on GPU 
for I in eachindex(A)
    A_sum += A[I]
end
A_sum

A'

CUDA.allowscalar(false) # disallowing it will disable printing of matrix on GPU
view(A', : , :)


A = CUDA.rand(1024)
R = sum(A; dims=1)
CUDA.@allowscalar R[]

# don't allow calling CPU code in GPU arrays
ccall(:whatever, Nothing, (Ptr{Float32},), CUDA.rand(1))


# %% [markdown]
# # Exercise

# %%
rmse(A,B) = sqrt(sum(abs2, A .- B) / length(A))
n = 1024
A = rand(n,n)
B = rand(n,n)
rmse(A,B)

dA = CuArray(A)
dB = CuArray(B)
rmse(dA, dB)

using BenchmarkTools
@benchmark rmse($A, $B)

@benchmark rmse($dA, $dB)


