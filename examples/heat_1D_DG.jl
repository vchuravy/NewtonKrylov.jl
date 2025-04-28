# # Implicit time-integration

# ## Necessary packages
using NewtonKrylov
using CairoMakie

include(joinpath(dirname(pathof(NewtonKrylov)), "..", "examples", "implicit.jl"))

using SummationByPartsOperators

xmin = 0.0
xmax = 1.0

polydeg = 3
elements = 40

D_local = legendre_derivative_operator(xmin = -1.0, xmax = 1.0, N = polydeg + 1)
mesh = UniformPeriodicMesh1D(; xmin, xmax, Nx = elements)

# D1m = D1p = couple_discontinuously(D_local, mesh, Val(:central))
D1m = couple_discontinuously(D_local, mesh, Val(:minus))
D1p = couple_discontinuously(D_local, mesh, Val(:plus))

using SparseArrays

D2 = sparse(D1m) * sparse(D1p)

x = grid(D1m)

# ## Heat 1D
# $ \frac{\partial u(x, t)}{\partial t} = a * \frac{\partial^2 u(x, t)}{\partial x^2 $

function heat_1D_v1!(du, u, (D2,), t)
    mul!(du, D2, u)
    return
end

function heat_1D!(du, u, (D1m, D1p), t)
    du1 = D1p * u
    # mul!(du, D1p, u)

    mul!(du, D1m, du1)
    return
end

# inital condition

f(x) = 4x * (1 - x)

using LinearAlgebra

# ## Investigate the Jacobian's


@show zero(x)

J = jacobian(G_Euler!, heat_1D_v1!, zero(x), (D2,), 0.1, 0.0)


# ### Euler
J = jacobian(G_Euler!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# Rank:

rank(J)

# Condition number

cond(Array(J))

# ### Midpoint

J = jacobian(G_Midpoint!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)
# Rank:

rank(J)

# Condition number

cond(Array(J))

# ### Trapezoid

J = jacobian(G_Trapezoid!, heat_1D!, zero(x), (D1m, D1p), 0.1, 0.0)

# Rank:

rank(J)

# Condition number

cond(Array(J))


function solve_heat_1D(G!, x, Δt, t_final, initial_condition, p)
    ts = 0.0:Δt:t_final

    u₀ = initial_condition.(x)

    hist = [copy(u₀)]
    callback = (u) -> push!(hist, copy(u))
    solve(G!, heat_1D!, u₀, p, Δt, ts; callback)

    return x, ts[1:length(hist)], hist
end


function plot_1D(xs, ts, hist)
    fig, ax = lines(xs, hist[1])
    for i in 2:length(hist)
        lines!(ax, xs, hist[i])
    end
    return fig
end

Δt = 0.01
t_final = 1.0
xs, ts, hist = solve_heat_1D(G_Euler!, x, Δt, t_final, f, (D1m, D1p));

lines(x, hist[end])

contour(xs, ts, stack(hist))
plot_1D(xs, ts, hist)

xs, ts, hist = solve_heat_1D(G_Midpoint!, x, Δt, t_final, f, (D1m, D1p));

lines(x, hist[end])

contour(xs, ts, stack(hist))
plot_1D(xs, ts, hist)

xs, ts, hist = solve_heat_1D(G_Trapezoid!, x, Δt, t_final, f, (D1m, D1p));

lines(x, hist[end])

contour(xs, ts, stack(hist))
plot_1D(xs, ts, hist)
