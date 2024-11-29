using NewtonKrylov
using CairoMakie

# 2D Climate Model 

function G!(res, f, y, yₙ, t, Δt)
    # F(y) = yₙ + Δt * f(y, t) - y
    res .= yₙ .+ Δt .* f(y, t) .- y
end

function f(x, t, γ)
    [ x[2],     # dx/dt = v 
    -γ^2* x[1]] # dv/dt = -γ^2 * x
end


function implicit_euler_spring()
    k = 2.    # spring constant
    m = 1.    # object's mass
    x0 = 0.1 # initial position
    v0 = 0.   # initial velocity
    
    t₀ = 0.0
    tₛ = 40.0
    Δt = 0.01
    
    ts = t₀:Δt:tₛ
    
    yₙ = [x0, v0]
    
    γ = sqrt(k/m)

    hist = [copy(yₙ)]
    for t in ts
        if t == t₀
            continue
        end
        F!(res, y) = G!(res, (y,t) -> f(y, t, γ), y, yₙ, t, Δt)
        y, _ = newton_krylov!(F!, copy(yₙ))
        push!(hist, y)
        yₙ .= y
    end
    hist, ts
end

hist, ts = implicit_euler_spring()
v = map(y -> y[1], hist)
x = map(y -> y[2], hist)


fig = Figure()
ax = fig[1,1]

lines(fig[1,1], ts, v)
lines(fig[1,2], ts, x)

fig