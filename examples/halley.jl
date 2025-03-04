## Simple 2D example from (Kelley2003)[@cite]

using NewtonKrylov, LinearAlgebra
using CairoMakie

function F!(res, x)
    res[1] = x[1]^2 + x[2]^2 - 2
    return res[2] = exp(x[1] - 1) + x[2]^2 - 2
end

function F(x)
    res = similar(x)
    F!(res, x)
    return res
end

import NewtonKrylov: Forcing, EisenstatWalker, inital, forcing, solve!,
                     JacobianOperator, HessianOperator, Stats, update, GmresSolver

function halley_krylov!(
        F!, u::AbstractArray, res::AbstractArray;
        tol_rel = 1.0e-6,
        tol_abs = 1.0e-12,
        max_niter = 50,
        forcing::Union{Forcing, Nothing} = EisenstatWalker(),
        verbose = 0,
        Solver = GmresSolver,
        M = nothing,
        N = nothing,
        krylov_kwargs = (;),
        callback = (args...) -> nothing,
    )
    t₀ = time_ns()
    F!(res, u) # res = F(u)
    n_res = norm(res)
    callback(u, res, n_res)

    tol = tol_rel * n_res + tol_abs

    if forcing !== nothing
        η = inital(forcing)
    end

    verbose > 0 && @info "Jacobian-Free Halley-Krylov" Solver res₀ = n_res tol tol_rel tol_abs η

    J = JacobianOperator(F!, res, u)
    H = HessianOperator(J)
    solver = Solver(J, res)
    
    stats = Stats(0, 0)
    while n_res > tol && stats.outer_iterations <= max_niter
        # Handle kwargs for Preconditoners
        kwargs = krylov_kwargs
        if N !== nothing
            kwargs = (; N = N(J), kwargs...)
        end
        if M !== nothing
            kwargs = (; M = M(J), kwargs...)
        end
        if forcing !== nothing
            # ‖F′(u)d + F(u)‖ <= η * ‖F(u)‖ Inexact Newton termination
            kwargs = (; rtol = η, kwargs...)
        end

        solve!(solver, J, copy(res); kwargs...) # J \ fx 
        a = copy(solver.x)

        # calculate hvvp (2nd order directional derivative using the JVP)
        hvvp = similar(res)
        mul!(hvvp, H, a)

        solve!(solver, J, hvvp; kwargs...) # J \ hvvp
        b = solver.x 

        # Update u
        @. u -= (a * a) / (a - b / 2) 

        # Update residual and norm
        n_res_prior = n_res

        F!(res, u) # res = F(u)
        n_res = norm(res)
        callback(u, res, n_res)

        if isinf(n_res) || isnan(n_res)
            @error "Inner solver blew up" stats
            break
        end

        if forcing !== nothing
            η = forcing(η, tol, n_res, n_res_prior)
        end

        verbose > 0 && @info "Newton" iter = n_res η=(forcing !== nothing ? η : nothing) stats
        stats = update(stats, solver.stats.niter) # TODO we do two calls to solver iterations
    end
    t = (time_ns() - t₀) / 1.0e9
    return u, (; solved = n_res <= tol, stats, t)
end

xs = LinRange(-3, 8, 1000)
ys = LinRange(-15, 10, 1000)

levels = [0.1, 0.25, 0.5:2:10..., 10.0:10:200..., 200:100:4000...]

fig, ax = contour(xs, ys, (x, y) -> norm(F([x, y])); levels)

trace_1 = let x₀ = [2.0, 0.5]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, n_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = newton_krylov!(F!, x₀, callback = hist)
    xs
end
lines!(ax, trace_1)

trace_2 = let x₀ = [2.0, 0.5]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, n_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = halley_krylov!(F!, x₀, similar(x₀), callback = hist, verbose=1, forcing=nothing)
    @show stats
    xs
end
lines!(ax, trace_2)

trace_2

fig
