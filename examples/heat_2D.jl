# # Implicit time-integration

# ## Necessary packages
using NewtonKrylov
using CairoMakie
using OffsetArrays

include("implicit.jl")
include("halovector.jl")

# ## Diffusion 2D

function diffusion!(du::HaloVector, u::HaloVector, p, t)
    return diffusion!(du.data, u.data, p, t)
end

function diffusion!(du, u, (a, Δx, Δy), _)
    N, M = size(u)
    N = N - 2
    M = M - 2

    # Enforce boundary conditions
    # (wrap around)
    u[0, :] .= u[N, :]
    u[N + 1, :] .= u[1, :]
    u[:, 0] .= u[:, N]
    u[:, N + 1] .= u[:, 1]

    for i in 1:N
        for j in 1:M
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

u₀ = HaloVector(OffsetArray(zeros(N + 2, M + 2), 0:(N + 1), 0:(M + 1)))

@edit -u₀

J = jacobian(G_Euler!, diffusion!, u₀, (a, Δx, Δy), Δt, 0.0)
J2 = jacobian(G_Midpoint!, diffusion!, u₀, (a, Δx, Δy), Δt, 0.0)
J3 = jacobian(G_Trapezoid!, diffusion!, u₀, (a, Δx, Δy), Δt, 0.0)

heatmap(0:(N + 1), 0:(N + 1), u₀.data.parent)

function create_video_implicit(filename, G!, f!, N, M, p, Δt, t_stop, frame_kwargs)
    xs = 0:(N + 1)
    ys = 0:(M + 1)

    u = HaloVector(OffsetArray(zeros(N + 2, M + 2), xs, ys))
    u.data.parent[N ÷ 2 .+ (-1:1), M ÷ 2 .+ (-1:1)] .= 5

    ts = 0.0:Δt:t_stop
    fig, ax, hm = heatmap(xs, ys, u.data.parent)
    return record(fig, filename; frame_kwargs...) do io
        function callback(_)
            Base.notify(hm.args[3])
            # autolimits!(ax) # update limits
            recordframe!(io)
            return yield()
        end
        solve(G!, f!, u, p, Δt, ts; callback, verbose = 2)
    end
end

create_video_implicit(
    joinpath(@__DIR__, "implicit_euler.mp4"),
    G_Euler!, diffusion!, N, M, (a, Δx, Δy), Δt, 4 * Δt, (; framerate = 1)
)


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


explicit_diffusion(zeros(N, M), 0.01, 0.01, 0.01, 0.2)
implicit_diffusion(G_Midpoint!, zeros(N, M), 0.01, 0.01, 0.01, 0.02)

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
