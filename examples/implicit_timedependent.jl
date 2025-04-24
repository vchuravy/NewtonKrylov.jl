# # Implicit time-integration

# ## Necessary packages
using NewtonKrylov
using CairoMakie

# ## Spring equations

function f!(du, u, (γ,), t)
    du[1] = u[2] # dx/dt = v
    du[2] = -γ^2 * u[1] # dv/dt = -γ^2 * x
    return nothing
end

# ## Implicit schemes

# ### Implicit Euler

function G_Euler!(res, uₙ, Δt, f!, du, u, p, t)
    f!(du, u, p, t)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

# ### Implicit Midpoint

function G_Midpoint!(res, uₙ, Δt, f!, du, u, p, t)
    # Use res for a temporary allocation (uₙ .+ u) ./ 2
    uuₙ = res
    uuₙ .= (uₙ .+ u) ./ 2
    f!(du, uuₙ, p, t + Δt / 2)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

# ### Implicit Trapezoid

function G_Trapezoid!(res, uₙ, Δt, f!, du, u, p, t)
    # Use res as the temporary
    duₙ = res
    f!(duₙ, uₙ, p, t)
    f!(du, u, p, t + Δt)

    res .= uₙ .+ (Δt / 2) .* (duₙ .+ du) .- u
    return nothing
end


# ## Non-adaptive time stepping

function solve(G!, f!, uₙ, p, Δt, ts; callback = _ -> nothing)
    u = copy(uₙ)
    du = similar(uₙ)
    res = similar(uₙ)
    F!(res, u, (uₙ, Δt, du, p, t)) = G!(res, uₙ, Δt, f!, du, u, p, t)

    for t in ts
        if t == first(ts)
            continue
        end
        _, stats = newton_krylov!(F!, u, (uₙ, Δt, du, p, t), res)
        # @show stats
        callback(u)
        uₙ .= u
    end
    return uₙ
end

function implicit_spring(G! = G_Euler!)
    k = 2.0    # spring constant
    m = 1.0    # object's mass
    x0 = 0.1 # initial position
    v0 = 0.0   # initial velocity

    t₀ = 0.0
    tₛ = 40.0
    Δt = 0.01

    ts = t₀:Δt:tₛ

    uₙ = [x0, v0]

    γ = sqrt(k / m)

    hist = [copy(uₙ)]
    callback = (u) -> push!(hist, copy(u))
    solve(G!, f!, uₙ, (γ,), Δt, ts, ; callback)
    return hist, ts
end

# ## Plots

hist_euler, ts_euler = implicit_spring(G_Euler!)
v_euler = map(y -> y[1], hist_euler)
x_euler = map(y -> y[2], hist_euler)

hist_midpoint, ts_midpoint = implicit_spring(G_Midpoint!)
v_midpoint = map(y -> y[1], hist_midpoint)
x_midpoint = map(y -> y[2], hist_midpoint)


hist_trapezoid, ts_trapezoid = implicit_spring(G_Trapezoid!)
v_trapezoid = map(y -> y[1], hist_trapezoid)
x_trapezoid = map(y -> y[2], hist_trapezoid)


fig = Figure()
ax = fig[1, 1]

lines(fig[1, 1], ts_euler, v_euler, label = "Euler")
lines(fig[1, 2], ts_euler, x_euler, label = "Euler")

lines(fig[2, 1], ts_midpoint, v_midpoint, label = "Midpoint")
lines(fig[2, 2], ts_midpoint, x_midpoint, label = "Midpoint")

lines(fig[3, 1], ts_trapezoid, v_trapezoid, label = "Trapezoid")
lines(fig[3, 2], ts_trapezoid, x_trapezoid, label = "Trapezoid")

fig


# ## Jacobian of various G

function jacobian(G!, f!, uₙ, p, Δt, t)
    u = copy(uₙ)
    du = similar(uₙ)
    res = similar(uₙ)
    F!(res, u, (uₙ, Δt, du, p, t)) = G!(res, uₙ, Δt, f!, du, u, p, t)

    J = NewtonKrylov.JacobianOperator(F!, res, u, (uₙ, Δt, du, p, t))
    return collect(J)
end


# ### Implicit Euler

jacobian(G_Euler!, f!, [0.1, 0.0], (sqrt(2.0 / 1.0),), 0.01, 0.0)

# ### Implicit Midpoint

jacobian(G_Midpoint!, f!, [0.1, 0.0], (sqrt(2.0 / 1.0),), 0.01, 0.0)

# ### Implicit Trapezoid

jacobian(G_Trapezoid!, f!, [0.1, 0.0], (sqrt(2.0 / 1.0),), 0.01, 0.0)


# ## Erik's example

struct MyParams{T}
    x::T
end

function erik_f(du, u, p, _)
    x = p.x

    for i in eachindex(x)
        x[i] = u[i]^2
    end

    return du .= u .+ x
end

u0 = [1.0, 0.5]

p = MyParams(similar(u0))

solve(G_Euler!, erik_f, u0, p, 0.1, 0.0:0.1:1.0)
