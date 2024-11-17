module NewtonKrylov

export newton_krylov, newton_krylov!

using Krylov
using LinearAlgebra, SparseArrays
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

function Base.collect(JOp::JacobianOperator)
    N,M = size(JOp)
	v = zeros(eltype(JOp), M)
	out = zeros(eltype(JOp), N)
    J = SparseMatrixCSC{eltype(v), Int}(undef, size(JOp)...)
    for j in 1:M
        out .= 0.0
        v .= 0.0
        v[j] = 1.0
        mul!(out, JOp, v)
        for i in 1:N
            if out[i] != 0
                J[i, j] = out[i]
            end
        end
    end
    J
end

##
# Newton-Krylov
##
import Base:@kwdef 

abstract type Forcing end
@kwdef struct Fixed <: Forcing
    η::Float64=0.1
end

function (F::Fixed)(args...)
    return F.η
end
inital(F::Fixed) = F.η

@kwdef struct EisenstatWalker <:Forcing
    η_max::Float64=0.999
    γ::Float64=0.9
end

# @assert η_max === nothing || 0.0 < η_max < 1.0 

"""
Compute the Eisenstat-Walker forcing term for n > 0
"""
function (F::EisenstatWalker)(η, tol, n_res, n_res_prior)
    η_res = F.γ * n_res^2 / n_res_prior^2
    # Eq 3.6
    if F.γ * η^2 <= 1//10
        η_safe = min(F.η_max, η_res)
    else
        η_safe = min(F.η_max, max(η_res, F.γ*η^2))
    end
    return min(F.η_max, max(η_safe, 1//2 * tol / n_res)) # Eq 3.5
end
inital(F::EisenstatWalker) = F.η_max

function newton_krylov(F, u₀, M::Int = length(u₀); kwargs...)
    F!(res, u) = (res .= F(u); nothing)
    newton_krylov!(F!, u₀, M; kwargs...)
end

function newton_krylov!(F!, u₀, M::Int = length(u₀); kwargs...)
    res = similar(u₀, M)
    newton_krylov!(F!, u₀, res; kwargs...)
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
  - `forcing`: Maximum forcing term for inexact Newton.
             If `nothing` an exact Newton method is used.   
"""
function newton_krylov!(F!, u, res;
                        tol_rel=1e-6,
                        tol_abs=1e-12,
                        max_niter = 50,
                        forcing::Union{Forcing,Nothing} = EisenstatWalker(),
                        verbose = 0,
                        Solver = CgSolver,
                        M = nothing,
                        N = nothing,
                        ldiv = false)
    F!(res, u) # res = F(u)
    n_res = norm(res)
    tol = tol_rel * n_res + tol_abs

    if forcing !== nothing
        η = inital(forcing)
    end

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" Solver res₀=n_res tol tol_rel tol_abs η 
    
    J = JacobianOperator(F!, res, u)
    solver = Solver(J, res)

    n_iter = 1
    while n_res > tol && n_iter <= max_niter
        # Handle kwargs for Preconditoners
        kwargs = (; ldiv, verbose=verbose-1)
        if N !== nothing
            kwargs = (; N = N(J), kwargs...)
        end
        if M !== nothing
            kwargs = (; M = M(J), kwargs...)
        end
        if forcing !== nothing
            # ‖F′(u)d + F(u)‖ <= η * ‖F(u)‖ Inexact Newton termination
            kwargs = (;rtol=η, kwargs...)
        end

        # Solve: Jx = -res
        # res is modifyed by J, so we create a copy `-res`
        # TODO: provide a temporary storage for `-res`
        solve!(solver, J, -res; kwargs...)

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

        if forcing !== nothing
            η = forcing(η, tol, n_res, n_res_prior)
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

end # module NewtonKrylov
