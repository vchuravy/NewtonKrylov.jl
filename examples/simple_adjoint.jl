## Simple 2D example from (Kelley2003)[@cite]

using NewtonKrylov, LinearAlgebra
using CairoMakie

function F!(res, x, p)
    res[1] = p[1] * x[1]^2 + p[2] * x[2]^2 - 2
    return res[2] = exp(p[1] * x[1] - 1) + p[2] * x[2]^2 - 2
    # return nothing
end

function F(x, p)
    res = similar(x)
    F!(res, x, p)
    return res
end

p = [1.0, 1.3, 1.0]

xs = LinRange(-3, 8, 1000)
ys = LinRange(-15, 10, 1000)

levels = [0.1, 0.25, 0.5:2:10..., 10.0:10:200..., 200:100:4000...]

fig, ax = contour(xs, ys, (x, y) -> norm(F([x, y], p)); levels)


x₀ = [2.0, 0.5]
x, stats = newton_krylov!((res, u) -> F!(res, u, p), x₀)
@assert stats.solved

# ## Adjoint setup
# Define x̂ to be a solution we would like discover the parameter of.

const x̂ = [1.0000001797004159, 1.0000001140397106]

# `g` is our target function measuring the distance
function g(x, p)
    return sum(abs2, x .- x̂)
end

using Enzyme
using Krylov

# function adjoint_with_primal(F!, G, u₀, p; kwargs...)
#     res = similar(u₀)
#     u, stats = newton_krylov!(F!, u₀, res; kwargs...)
#     # @assert stats.solved

#     return (; u, loss = G(u, p), dp = adjoint_nl!(F!, G, res, u, p))
# end

"""
    adjoint_nl!(F!, G, res, u, p)

# Arguments
- `F!` -> F!(res, u, p) solves F(u; p) = 0
- `G`  -> "Target function"/ "Loss function" G(u, p) = scalar
"""
function adjoint_nl!(F!, G, res, u, p)
    # Calculate gₚ and gₓ
    gₚ = Enzyme.make_zero(p)
    gₓ = Enzyme.make_zero(u)
    Enzyme.autodiff(Enzyme.Reverse, G, Duplicated(u, gₓ), Duplicated(p, gₚ))

    # Solve adjoint equation for λ
    J = NewtonKrylov.JacobianOperator((res, u) -> F!(res, u, p), res, u)
    λ, stats = gmres(transpose(J), gₓ)
    @assert stats.solved

    # Now do vJp for λᵀ*fₚ
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(
        Enzyme.Reverse, F!, Const,
        Duplicated(res, λ),
        Const(u),
        Duplicated(p, dp)
    )

    # TODO:
    # Use recursive_map to implement this subtraction https://github.com/EnzymeAD/Enzyme.jl/pull/1852
    return gₚ - dp
end

adjoint_with_primal(F!, g, x₀, p)

# ## TODO:
# Use Optimizer.jl to find `p`
