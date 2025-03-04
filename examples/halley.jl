## Simple 2D example from (Kelley2003)[@cite]

using NewtonKrylov, LinearAlgebra
using CairoMakie
using Krylov, Enzyme

function F!(res, x)
    res[1] = x[1]^2 + x[2]^2 - 2
    return res[2] = exp(x[1] - 1) + x[2]^2 - 2
end

function F(x)
    res = similar(x)
    F!(res, x)
    return res
end

function halley(F!, u, res;
        tol_rel = 1.0e-6,
        tol_abs = 1.0e-12,
        max_niter = 50,
        Solver = GmresSolver,
    )

    F!(res, u) # res = F(u)
    n_res = norm(res)

    tol = tol_rel * n_res + tol_abs

    J = NewtonKrylov.JacobianOperator(F!, res, u)
    H = NewtonKrylov.HessianOperator(J)
    solver = Solver(J, res)
    
    for i in :max_niter
        if n_res <= tol
            break
        end
        solve!(solver, J, copy(res)) # J \ fx 
        a = copy(solver.x)

        # calculate hvvp (2nd order directional derivative using the JVP)
        hvvp = similar(res)
        mul!(hvvp, H, a)

        solve!(solver, J, hvvp) # J \ hvvp
        b = solver.x 

        # update 
        @. u -= (a * a) / (a - b / 2) 

    end
end

# u = [2.0, 0.5]
# res = zeros(2)
# J = NewtonKrylov.JacobianOperator(F!,u,res)
# F!(res, u)
# a, stats = gmres(J, copy(res))

# J_cache = Enzyme.make_zero(J)
# out = similar(J.res)
# hvvp = Enzyme.make_zero(out)
# du = Enzyme.make_zero(J.u)
# autodiff(Forward, LinearAlgebra.mul!, 
#     DuplicatedNoNeed(out, hvvp), 
#     DuplicatedNoNeed(J, J_cache), 
#     DuplicatedNoNeed(du, a))

# hvvp

# b, stats = gmres(J, hvvp)
# @. u -= (a * a) / (a - b / 2) 

# a


dg_ad(x, dx) = autodiff(Forward, flux, DuplicatedNoNeed(x, dx))[1]
ddg_ad(x, dx, ddx) = autodiff(Forward, dg_ad, DuplicatedNoNeed(x, dx),
                              DuplicatedNoNeed(dx, ddx))[1]

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

trace_2 = let x₀ = [2.5, 3.0]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, n_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = newton_krylov!(F!, x₀, callback = hist)
    xs
end
lines!(ax, trace_2)

trace_3 = let x₀ = [3.0, 4.0]
    xs = Vector{Tuple{Float64, Float64}}(undef, 0)
    hist(x, res, n_res) = (push!(xs, (x[1], x[2])); nothing)
    x, stats = newton_krylov!(F!, x₀, callback = hist, forcing = NewtonKrylov.EisenstatWalker(η_max = 0.68949), verbose = 1)
    @show stats.solved
    xs
end
lines!(ax, trace_3)

fig
