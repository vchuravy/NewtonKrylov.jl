# # Implicit time-integration

# ## Necessary packages
using NewtonKrylov
using CairoMakie
using ComponentArrays

import Enzyme

Enzyme.Duplicated(CA::ComponentArray{T}, dA::Array{T}) where {T} = Enzyme.Duplicated(CA, ComponentArray(dA, getfield(CA, :axes)))

include("implicit.jl")

# ## Diffusion 2D

function diffusion!(du::ComponentVector, u::ComponentVector, p, t)
    return diffusion!(du.u, u.u, p, t)
end

# Centered finite difference

function diffusion!(du, u, (a, Δx, Δy), _)
    N, M = size(u)

    # Enforce boundary conditions
    # (wrap around)
    u[1, :] .= u[N - 1, :]
    u[N, :] .= u[2, :]
    u[:, 1] .= u[:, N - 1]
    u[:, N] .= u[:, 2]

    du[1, :] .= 0
    du[N, :] .= 0
    du[:, 1] .= 0
    du[:, N] .= 0

    for i in 2:(N - 1)
        for j in 2:(M - 1)
            du[i, j] = a * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / Δx^2 +
                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / Δy^2
            )
        end
    end
    return
end

a = 0.01

N = M = 40

Δx = 1 / (N + 1)  # x-grid spacing
Δy = 1 / (M + 1)  # y-grid spacing

# TODO: This is a time-step from explicit
Δt = Δx^2 * Δy^2 / (2.0 * a * (Δx^2 + Δy^2)) # Largest stable time step

J = jacobian(G_Euler!, diffusion!, zeros(4, 4), (a, Δx, Δy), Δt, 0.0) |> Array
J2 = jacobian(G_Midpoint!, diffusion!, zeros(4, 4), (a, Δx, Δy), Δt, 0.0) |> Array
J3 = jacobian(G_Trapezoid!, diffusion!, zeros(4, 4), (a, Δx, Δy), Δt, 0.0) |> Array

NewtonKrylov.JacobianOperator((du, u, p) -> diffusion!(du, u, p, 0.0), zeros(4, 4), zeros(4, 4), (a, Δx, Δy)) |> collect |> Array

using CairoMakie
function explicit_diffusion(u₀, a, Δx, Δy, t_stop; callback = (u, t) -> nothing)
    N, M = size(u₀)
    Δt = Δx^2 * Δy^2 / (2.0 * a * (Δx^2 + Δy^2)) # Largest stable time step

    u = u₀
    du = zero(u)

    u[N ÷ 2 .+ (-1:1), M ÷ 2 .+ (-1:1)] .= 5

    t = 0.0
    callback(u, t)

    while t < t_stop
        t += Δt
        diffusion!(du, u, (a, Δx, Δy), t)
        u .+= Δt .* du
        callback(u, t)
    end
    return u
end

function implicit_diffusion(G!, u₀, a, Δx, Δy, t_stop; callback = (u, t) -> nothing)
    N, M = size(u₀)

    @show Δt = Δx^2 * Δy^2 / (2.0 * a * (Δx^2 + Δy^2)) # Largest stable time step

    u = u₀
    du = zero(u)

    u[N ÷ 2 .+ (-1:1), M ÷ 2 .+ (-1:1)] .= 5

    # ComponentVector since Krylov.jl expects `b` to be a Vector...
    u = ComponentVector(u = u)
    du = ComponentVector(u = du)

    res = similar(u)
    uₖ = zero(u)

    t = 0.0
    callback(u, t)

    p = (a, Δx, Δy)

    F!(res, uₖ, (u, Δt, du, p, t)) = G!(res, u, Δt, diffusion!, du, uₖ, p, t)
    while t < t_stop
        t += Δt
        _, stats = newton_krylov!(F!, uₖ, (u, Δt, du, p, t), res)
        @show stats
        # @show stats
        u .= uₖ
        callback(u, t)
        if !stats.solved
            break
        end
    end
    return u.u
end


explicit_diffusion(zeros(N, M), 0.01, 0.01, 0.01, 0.2)
implicit_diffusion(G_Midpoint!, zeros(N, M), 0.01, 0.01, 0.01, 0.02)

using CairoMakie

function create_video(filename, method, N, M, method_args, frame_kwargs)
    xs = 1:N
    ys = 1:M

    u = zeros(N, M)

    fig, ax, hm = heatmap(xs, ys, u)
    return record(fig, filename; frame_kwargs...) do io
        function callback(_u, t)
            Base.notify(hm.args[3])
            # autolimits!(ax) # update limits
            recordframe!(io)
            return yield()
        end
        method(u, method_args...; callback)
    end
end

create_video(joinpath(@__DIR__, "explicit.mp4"), explicit_diffusion, N, M, (0.01, 0.01, 0.01, 0.2), (; framerate = 10))
# create_video(joinpath(@__DIR__, "implicit_euler.mp4"), (args...; kwargs...)->implicit_diffusion(G_Euler!, args...; kwargs...), 40, 40, (0.01, 0.01, 0.01, 0.2), (;framerate=10))
# create_video(joinpath(@__DIR__, "implicit_midpoint.mp4"), (args...; kwargs...)->implicit_diffusion(G_Midpoint!, args...; kwargs...), 40, 40, (0.01, 0.01, 0.01, 0.2), (;framerate=10))
# create_video(joinpath(@__DIR__, "implicit_trapezoid.mp4"), (args...; kwargs...)->implicit_diffusion(G_Trapezoid!, args...; kwargs...), 40, 40, (0.01, 0.01, 0.01, 0.2), (;framerate=10))
