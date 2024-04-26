# %% [markdown]
# # How to properly Benchmark


# %%
using Pkg; Pkg.activate(dirname(dirname(@__FILE__)))
using CUDA, BenchmarkTools
CUDA.allowscalar(false)

n = 32 
A = CUDA.rand(Float32,n,n)
B = CUDA.rand(Float32,n,n)

# warmup
A * B

@time A * B

# ask for CPU to wait for GPU to finish computing before it access the "results"
synchronize() # ensure GPU is idle

@time begin
    A * B # do computation
    synchronize() # wait for GPU to come to a "stop"
end

synchronize()
@time CUDA.@sync A * B

CUDA.@time A * B # includes synchronize in this macro

# for reusing the previously ran benchmark
CUDA.@timed A * B

# returns the number of seconds a computation took
CUDA.@elapsed A * B

# %% [markdown]
# # More robust benchamrk 

# %%
A = CUDA.rand(Float32, n ,n)
B = CUDA.rand(Float32, n ,n)

@benchmark CUDA.@sync A * B

using Profile
@profile CUDA.@sync A * B
Profile.print(; noisefloor=30.) # no info about actual computation, all on data loading from CPU to GPu

# use this! 
# the higher the usage of GPU the better
CUDA.@profile A * B

GC.gc(true)
CUDA.reclaim()
CUDA.@bprofile A * B

CUDA.@profile trace=true A * B


rmse(A,B) = sqrt(sum((A-B).^2) / length(A))

A = CUDA.rand(64,64)
B = CUDA.rand(64,64)

# why the out of memory error?
CUDA.@bprofile rmse(A,B)


# broadcast over operation -, to reduce the number of kernel calls
rmse(A,B) = sqrt(sum((A.-B).^2) / length(A))


CUDA.@bprofile rmse(A,B)

# go even further

# this could be automated with LazyArrays.jl with @- macro 
function rmse(A,B)
    bc = Base.broadcasted(A,B) do a,b
        (a - b)^2
    end
    sqrt(sum(bc) / length(A))
end


CUDA.@bprofile rmse(A,B)

# Use Nsight system
# CUDA.@profile macro,passing external=true;

# NVTX: NVIDIA Tools Extensions
# you may install it those on mac and use it to profile your code running on a server




N = 16 
A = CUDA.rand(2048,2048,N)
B = CUDA.rand(2048,2048,N)

using NVTX
NVTX.@annotate function rmse(A,B)
    bc = Base.broadcasted(A,B) do a,b
        (a - b)^2
    end
    bc = Broadcast.instantiate(bc)
    sqrt(sum(bc) / length(A))


end

NVTX.@annotate function doit()
    rmses = Vector{eltype(A)}(undef, N)
    for i in 1:N
        rmses[i] = rmse(A[:,:,i], B[:,:,i])
    end
    rmses
end

@benchmark doit()

CUDA.@profile external=true (doit(); doit());


NVTX.@annotate function doit()
    rmses = Vector{eltype(A)}(undef, N)
    for i in 1:N
        rmses[i] = @views rmse(A[:,:,i], B[:,:,i])
    end
    rmses
end

CUDA.@profile  (doit(); doit());


# avoids synchronization by reducing 

NVTX.@annotate function rmse(A,B)
    bc = Base.broadcasted(A,B) do a, b
        (a - b)^2
    end
    bc = Broadcast.instantiate(bc)
    sqrt.(sum(bc; dims=(1,2)) ./ length(A))
end

NVTX.@annotate function doit()
    rmses = Vector(undef, N)
    for i in 1:N
        rmses[i] = @views rmse(A[:,:,i], B[:,:,i])
    end
    vcat(Array.(rmses)...)
end

@benchmark doit()

CUDA.@profile external=false (doit();doit());

NVTX.@annotate function rmse!(C,A,B)
    bc = Base.broadcasted(A,B) do a,b
        (a - b)^2
    end
    bc = Broadcast.instantiate(bc)
    Base.mapreducedim!(identity, + , C, bc)
    C .= sqrt.(C ./ length(A))
    return
end

CUDA.allowscalar(true)
NVTX.@annotate function doit()
    rmses = similar(A,N)
    fill!(rmses, zero(eltype(A)))
    for i in 1:N
        @views rmse!(reshape(rmses[i],1,1), A[:,:,i], B[:,:,i])
    end
    Array(rmses)
end

doit()