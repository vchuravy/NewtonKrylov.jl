module NewtonKrylov

export newton_krylov, newton_krylov!

using Krylov
using LinearAlgebra
using Enzyme

##
# JacobianOperator
## 
import LinearAlgebra: mul!

function maybe_duplicated(f,df)
    if Enzyme.Compiler.active_reg(typeof(f))
        return DuplicatedNoNeed(f,df)
    else
        return Const(f)
    end
end

struct JacobianOperator{F, A}
    f::F # F!(res, u)
    f_cache::F
    res::A
    u::A
    function JacobianOperator(f::F, res, u) where F
        f_cache = Enzyme.make_zero(f)
        new{F, typeof(u)}(f, f_cache, res, u)
    end
end

Base.size(J::JacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::JacobianOperator) = eltype(J.u)

function mul!(out, J::JacobianOperator, v)
    Enzyme.make_zero!(J.f_cache)
    autodiff(Forward, 
        maybe_duplicated(J.f, J.f_cache), Const, 
        DuplicatedNoNeed(J.res, out), DuplicatedNoNeed(J.u, v)
    )
    nothing
end

function newton_krylov(F, u₀, M::Int = length(u₀); kwargs...)
    F!(res, u) = (res .= F(u); nothing)
    newton_krylov!(F!, u₀, M; kwargs...)
end

function newton_krylov!(F!, u₀, M::Int = length(u₀); kwargs...)
    res = similar(u₀, M)
    newton_krylov!(F!, u₀, res; kwargs...)
end

# TODO: LinearAlgebra.mul!(out, transpose(J), v)

##
# Newton-Krylov
##

"""
forcing(η, η_max, tol, n_res, n_res_prior)

Compute the Eisenstat-Walker forcing term for n > 0
"""
function forcing(η, η_max, tol, n_res, n_res_prior, γ=0.9)
    η_res = γ * n_res^2 / n_res_prior^2
    # Eq 3.6
    if γ * η^2 <= 1//10
        η_safe = min(η_max, η_res)
    else
        η_safe = min(η_max, max(η_res, γ*η^2))
    end
    return min(η_max, max(η_safe, 1//2 * tol / n_res)) # Eq 3.5
end

"""
"""
function newton_krylov!(F!, u, res;
                        tol_rel=1e-6,
                        tol_abs=1e-12,
                        max_niter = 50,
                        η_max = 0.9999,
                        verbose = false)
    J = JacobianOperator(F!, res, u)
    solver = CgSolver(size(J)..., typeof(u))

    F!(res, u) # res = F(u)
    n_res = norm(res)
    tol = tol_rel * n_res + tol_abs

    @assert 0.0 < η_max < 1.0
    η = η_max

    verbose && @info "Jacobian-Free Newton-Krylov" n_res tol tol_rel tol_abs η
    n_iter = 1
	while n_res > tol && n_iter <= max_niter
        # Solve: Jx = -res
        # res is modifyed by J
		solve!(solver, J, -res; rtol=η)

        verbose && @show solver.stats
        d = solver.x # Newton direction
        s = 1        # Newton step TODO: LineSearch

        # Update u
        u .+= s .* d

        # Update residual and norm
        F!(res, u) # res = F(u)
        n_res_prior = n_res
        n_res = norm(res)

        η = forcing(η, η_max, tol, n_res, n_res_prior)

        verbose && @info "Newton" iter=n_iter n_res η
        n_iter += 1
	end
    u
end

# TODO:
# - Better statistic
# - Allow choice of Krylov solver
#   - maxit_krylov?
# - Preconditoners
#  - See 3.3


end # module NewtonKrylov
