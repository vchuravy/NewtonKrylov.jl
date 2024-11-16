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
"""
function newton_krylov!(F!, u, res;
                        tol_rel=1e-6,
                        tol_abs=1e-12,
                        max_niter = 50,
                        verbose = false)
    J = JacobianOperator(F!, res, u)
    solver = CgSolver(size(J)..., typeof(u))

    F!(res, u) # res = F(u)
    n_res = norm(res)
    tol = tol_rel * n_res + tol_abs


    verbose && @info "Jacobian-Free Newton-Krylov" n_res tol tol_rel tol_abs
    n_iter = 1
	while n_res > tol && n_iter <= max_niter
        # Solve: Jx = -res
		solve!(solver, J, -res)

        d = solver.x # Newton direction
        s = 1        # Newton step

        # Update u
        u .+= s .* d

        # Update residual and norm
        F!(res, u) # res = F(u)
        n_res = norm(res)

        verbose && @info "Newton" iter=n_iter n_res tol
        n_iter += 1
	end
    u
end

# TODO:
# - Better statistic
# - Allow choice of Krylov solver
#   - maxit_krylov?
# - Inexact Netwon-Krylov
#   - First constant η = 0.1
#   - Eisenstat-Walker 
#   - 3.4.3
# - Preconditoners
#  - See 3.3


end # module NewtonKrylov
