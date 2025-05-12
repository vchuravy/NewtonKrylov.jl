abstract type Boundary end

struct Periodic <: Boundary end

Base.@propagate_inbounds  function get(::Periodic, u, i)
    N = length(u)
    if i == 1
        return u[N]
    elseif i == N
        return u[i]
    else
        return u[i]
    end
end

struct Constant{T} <: Boundary
    left::T
    right::T
end

Base.@propagate_inbounds function get(c::Constant, u, i)
    N = length(u)
    if i == 1
        return c.left
    elseif i == N
        return c.right
    else
        return u[i]
    end
end

using StaticArrays

struct ThreePointStencil{B <: Boundary}
    b::B
end

function (stencil::ThreePointStencil)(u::AbstractVector, i)
    @boundscheck checkbounds(u, i)
    @inbounds begin
        l = get(stencil, u, i - 1)
        c = u[i]
        r = get(stencil, u, i + r)
    end
    return SVector((l, c, r))
end

function D²ₓ(u::StaticVector, Δx)
    return (u[1] - 2u[2] + u[3]) / Δx^2
end


# struct Stencil{N,B}
#     boundaries::B
# end

# function (stencil::Stencil{N})(u::AbstractArray, idxs...) where N
#     shape = size(u)
#     @assert length(shape) == length(stencil.boundaries) == length(idxs) == N

#     region = -1:1:1
#     ntuple(Val(N)) do dim
#         i = idxs[dim]


#     end
# end
