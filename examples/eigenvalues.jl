using NewtonKrylov
using Arpack

using SparseArrays, LinearAlgebra
using CairoMakie

# ## 1D Bratu equations
# $y′′ + λ * exp(y) = 0$

# F(u) = 0

function bratu!(res, y, Δx, λ)
    N = length(y)
    for i in 1:N
        y_l = i == 1 ? zero(eltype(y)) : y[i - 1]
        y_r = i == N ? zero(eltype(y)) : y[i + 1]
        y′′ = (y_r - 2y[i] + y_l) / Δx^2

        res[i] = y′′ + λ * exp(y[i]) # = 0
    end
    return nothing
end

function bratu(y, dx, λ)
    res = similar(y)
    bratu!(res, y, dx, λ)
    return res
end

# ## Choice of parameters
const N = 500
const λ = 3.51382
const dx = 1 / (N + 1) # Grid-spacing

# ### Domain and Inital condition
x = LinRange(0.0 + dx, 1.0 - dx, N)
u₀ = sin.(x .* π)

JOp = NewtonKrylov.JacobianOperator(
    (res, u) -> bratu!(res, u, dx, λ),
    similar(u₀),
    copy(u₀)
)

# Q: is this generally true
LinearAlgebra.issymmetric(::NewtonKrylov.JacobianOperator) = true

# issymmetric(collect(JOp))

l, ϕ = eigs(JOp; nev = 10)
l2, ϕ2 = eigs(collect(JOp); nev = 10)


@time  eigs(collect(JOp); nev = 300)
@time  eigs(JOp; nev = 300)