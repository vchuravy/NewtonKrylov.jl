# # Implicit time-integration

# ## Necessary packages
using NewtonKrylov
using CairoMakie

# ## Implicit schemes

# ### Implicit Euler

function G_Euler!(res, f, y, yₙ, t, Δt)
    return res .= yₙ .+ Δt .* f(y, t) .- y
end

# ### Implicit Midpoint

function G_Midpoint!(res, f, y, yₙ, t, Δt)
    return res .= yₙ .+ Δt .* f((yₙ .+ y) ./ 2, t + Δt / 2) .- y
end

# ### Implicit Trapezoid

function G_Trapezoid!(res, f, y, yₙ, t, Δt)
    return res .= yₙ .+ (Δt / 2) .* (f(yₙ, t) .+ f(y, t + Δt)) .- y
end

# ## Spring equations

function f(x, t, γ)
    return [
        x[2],     # dx/dt = v
        -γ^2 * x[1],
    ] # dv/dt = -γ^2 * x
end

# ## Non-adaptive time stepping

function implicit_spring(G! = G_Euler!)
    k = 2.0    # spring constant
    m = 1.0    # object's mass
    x0 = 0.1 # initial position
    v0 = 0.0   # initial velocity

    t₀ = 0.0
    tₛ = 40.0
    Δt = 0.01

    ts = t₀:Δt:tₛ

    yₙ = [x0, v0]

    γ = sqrt(k / m)

    hist = [copy(yₙ)]
    for t in ts
        if t == t₀
            continue
        end
        F!(res, y, (yₙ, t, Δt)) = G!(res, (y, t) -> f(y, t, γ), y, yₙ, t, Δt)
        y, _ = newton_krylov!(F!, copy(yₙ), (yₙ, t, Δt))
        push!(hist, y)
        yₙ .= y
    end
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

lines(fig[3, 1], ts_trapezoid, v_trapezoid, label = "Midpoint")
lines(fig[3, 2], ts_trapezoid, x_trapezoid, label = "Midpoint")

fig
