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

LinearAlgebra.adjoint(J::JacobianOperator) = Adjoint(J)
LinearAlgebra.transpose(J::JacobianOperator) = Transpose(J)

# Jᵀ(y, u) = ForwardDiff.gradient!(y, x -> dot(F(x), u), xk)
# or just reverse mode

function mul!(out, J′::Union{Adjoint{<:Any, <:JacobianOperator}, Transpose{<:Any, <:JacobianOperator}}, v)
    J = parent(J′)
    Enzyme.make_zero!(J.f_cache)
    # TODO: provide cache for `copy(v)`
    # Enzyme zeros input derivatives and that confuses the solvers.
    autodiff(Reverse,
        maybe_duplicated(J.f, J.f_cache), Const,
        DuplicatedNoNeed(J.res, copy(v)), DuplicatedNoNeed(J.u, out)
    )
    nothing
end

##
# Newton-Krylov
##

function newton_krylov(F, u₀, M::Int = length(u₀); kwargs...)
    F!(res, u) = (res .= F(u); nothing)
    newton_krylov!(F!, u₀, M; kwargs...)
end

function newton_krylov!(F!, u₀, M::Int = length(u₀); kwargs...)
    res = similar(u₀, M)
    newton_krylov!(F!, u₀, res; kwargs...)
end

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

## Arguments
  - `F!`: `F!(res, u)` solves `res = F(u) = 0`
  - `u`: Initial guess
  - `res`: Temporary for residual
## Keyword Arguments
  - `tol_rel`: Relative tolerance
  - `tol_abs`: Absolute tolerance
  - `max_niter`: Maximum number of iterations
  - `η_max`: Maximum forcing term for inexact Newton.
             If `nothing` an exact Newton method is used.   
"""
function newton_krylov!(F!, u, res;
                        tol_rel=1e-6,
                        tol_abs=1e-12,
                        max_niter = 50,
                        η_max::Union{Real,Nothing} = 0.9999,
                        verbose = 0,
                        Solver = CgSolver)
    F!(res, u) # res = F(u)
    n_res = norm(res)
    tol = tol_rel * n_res + tol_abs

    @assert η_max === nothing || 0.0 < η_max < 1.0 
    if η_max === nothing
        η = √eps(eltype(u))
    else
        η = η_max
    end

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" Solver res₀=n_res tol tol_rel tol_abs η 
    
    J = JacobianOperator(F!, res, u)
    solver = Solver(J, res)

    n_iter = 1
    while n_res > tol && n_iter <= max_niter
        # Solve: Jx = -res
        # res is modifyed by J, so we create a copy `-res`
        # TODO: provide a temporary storage for `-res`
        solve!(solver, J, -res; rtol=η, verbose=verbose-1)

        d = solver.x # Newton direction
        s = 1        # Newton step TODO: LineSearch

        # Update u
        u .+= s .* d

        # Update residual and norm
        F!(res, u) # res = F(u)
        n_res_prior = n_res
        n_res = norm(res)

        if isinf(n_res) || isnan(n_res)
            error("Inner solver blew up at iter=$n_iter")
        end

        if η_max !== nothing
            η = forcing(η, η_max, tol, n_res, n_res_prior)
        end

        verbose > 0 && @info "Newton" iter=n_iter n_res η
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
