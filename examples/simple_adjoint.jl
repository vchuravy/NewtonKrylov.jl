## Simple 2D example from (Kelley2003)[@cite]

using NewtonKrylov, LinearAlgebra
using CairoMakie

function F!(res, x, p)
    res[1] = p[1] * x[1]^2 + p[2] * x[2]^2 - 2
    return res[2] = exp(p[1] * x[1] - 1) + p[2] * x[2]^2 - 2
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
x, stats = newton_krylov((u) -> F(u, p), x₀)

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

Jᵀ = Enzyme.jacobian(Enzyme.Reverse, u -> F(u, p), x) |> only

q = Jᵀ \ gₓ

gₚ - reshape(transpose(q) * Fₚ, :)

function everything_all_at_once(p)
    x₀ = [2.0, 0.5]
    x, _ = newton_krylov((u) -> F(u, p), x₀)
    return g(x, p)
end

everything_all_at_once(p)
Enzyme.gradient(Enzyme.Reverse, everything_all_at_once, p)
