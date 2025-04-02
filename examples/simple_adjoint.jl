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

const x̂ = [1.0000001797004159, 1.0000001140397106]

function g(x, p)
    return sum(abs2, x .- x̂)
end

g(x, p)

using Enzyme

function dg_p(x, p)
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(Enzyme.Reverse, g, Const(x), Duplicated(p, dp))
    return dp
end

gₚ = dg_p(x, p)

function dg_x(x, p)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Enzyme.Reverse, g, Duplicated(x, dx), Const(p))
    return dx
end

gₓ = dg_x(x, p)

Fₚ = Enzyme.jacobian(Enzyme.Reverse, p -> F(x, p), p) |> only

J = Enzyme.jacobian(Enzyme.Reverse, u -> F(u, p), x) |> only

q = transpose(J) \ gₓ

@show gₓ
display(transpose(J))
q2, stats = gmres(transpose(J), gₓ)
@assert q == q2

gₚ - transpose(Fₚ) * q


# dp = vJp_p(F!, res, x, p, q)


# F!(res, x, p); res = 0  || F(x,p) = 0
# function vJp_x(F!, res, x, p, v)
#     dx = Enzyme.make_zero(x)
#     Enzyme.autodiff(Enzyme.Reverse, F!,
#                     DuplicatedNoNeed(res, reshape(v, size(res))),
#                     Duplicated(x, dx),
#                     Const(p))
#     dx
# end

# function vJp_p(F!, res, x, p, v)
#     dp = Enzyme.make_zero(p)
#     Enzyme.autodiff(Enzyme.Reverse, F!,
#                     DuplicatedNoNeed(res, reshape(v, size(res))),
#                     Const(x),
#                     Duplicated(p, dp))
#     dp
# end


# Notes: "discretise-then-optimise", or "backpropagation through the solver" has the benefit of only requiring "resursive accumulate" on the shadow
#        whereas "continous adjoint" after SGJ notes requires parameters to be "vector" like.

# function everything_all_at_once(p)
#     x₀ = [2.0, 0.5]
#     x, _ = newton_krylov((u) -> F(u, p), x₀)
#     return g(x, p)
# end

# everything_all_at_once(p)
# Enzyme.gradient(Enzyme.Reverse, everything_all_at_once, p)

# struct JacobianOperatorP{F, A}
#     f::F # F!(res, u, p)
#     res::A
#     u::A
#     p::AbstractArray
#     function JacobianOperatorP(f::F, res, u, p) where {F}
#         return new{F, typeof(u), typeof(p)}(f, res, u, p)
#     end
# end

# Base.size(J::JacobianOperatorP) = (length(J.res), length(J.p))
# Base.eltype(J::JacobianOperatorP) = eltype(J.u)

# function mul!(out, J::JacobianOperatorP, v)
#     # Enzyme.make_zero!(J.f_cache)
#     f_cache = Enzyme.make_zero(J.f) # Stop gap until we can zero out mutable values
#     autodiff(
#         Forward,
#         maybe_duplicated(J.f, f_cache), Const,
#         DuplicatedNoNeed(J.res, reshape(out, size(J.res))),
#         Const(J.u),
#         # DuplicatedNoNeed(J.u, Enzyme.make_zero(J.u)) #, reshape(v, size(J.u)))
#     )
#     return nothing
# end


# #####

# function dg(x,dx, y, dy)
#     _x = x[1]
#     _y = y[1]
#     _a = _x * _y

#     x[1] = _a
#     #
#     _da = 0
#     _da += dx[1]
#     dx[1] = 0

#     _dx = 0
#     _dx += _da*_y

#     _dy = 0
#     _dy += _da*_x

#     _da = 0

#     dy[1] += _dy
#     _dy = 0
#     dx[1] += _dx 3-element Vector{Float64}:

# function f(x, y)
#     g(x, y)
#     g(x, y)
# end

# x = [1.0]
# y = [1.0]
# dx = [1.0]
# dy = [0.0]

# autodiff(Enzyme.Reverse, f, Duplicated(x, dx), Duplicated(y, dy))

dx
dy

using Enzyme
using Krylov

function adjoint_with_primal(F!, G, u₀, p; kwargs...)
    res = similar(u₀)
    # TODO: Adjust newton_krylov interface to work with `F(u, p)`
    u, stats = newton_krylov!((res, u) -> F!(res, u, p), u₀, res; kwargs...)
    # @assert stats.solved

    return (; u, loss = G(u, p), dp = adjoint_nl!(F!, G, res, u, p))
end

"""
    adjoint_nl!(F!, G, )

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
    λ, stats = gmres(transpose(J), gₓ) # todo why no transpose(gₓ)
    @assert stats.solved

    # Now do vJp for λᵀ*fₚ
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(
        Enzyme.Reverse, F!, Const,
        DuplicatedNoNeed(res, λ),
        Const(u),
        DuplicatedNoNeed(p, dp)
    )

    return gₚ - dp
end

adjoint_nl!(F!, g, similar(x), x, p)

adjoint_with_primal(F!, g, x₀, p)
