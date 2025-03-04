module NewtonKrylov

export newton_krylov, newton_krylov!

using Krylov
using LinearAlgebra, SparseArrays
using Enzyme

##
# JacobianOperator
##
import LinearAlgebra: mul!

function maybe_duplicated(f, df)
    if !Enzyme.Compiler.guaranteed_const(typeof(f))
        return DuplicatedNoNeed(f, df)
    else
        return Const(f)
    end
end

"""
    JacobianOperator

Efficient implementation of `J(f,x) * v` and `v * J(f, x)'`
"""
struct JacobianOperator{F, A}
    f::F # F!(res, u)
    f_cache::F
    res::A
    u::A
    function JacobianOperator(f::F, res, u) where {F}
        f_cache = Enzyme.make_zero(f)
        return new{F, typeof(u)}(f, f_cache, res, u)
    end
end

Base.size(J::JacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::JacobianOperator) = eltype(J.u)

function mul!(out, J::JacobianOperator, v)
    # Enzyme.make_zero!(J.f_cache)
    f_cache = Enzyme.make_zero(J.f) # Stop gap until we can zero out mutable values
    autodiff(
        Forward,
        maybe_duplicated(J.f, f_cache), Const,
        DuplicatedNoNeed(J.res, reshape(out, size(J.res))),
        DuplicatedNoNeed(J.u, reshape(v, size(J.u)))
    )
    return nothing
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
    autodiff(
        Reverse,
        maybe_duplicated(J.f, J.f_cache), Const,
        DuplicatedNoNeed(J.res, reshape(copy(v), size(J.res))),
        DuplicatedNoNeed(J.u, reshape(out, size(J.u)))
    )
    return nothing
end

function Base.collect(JOp::JacobianOperator)
    N, M = size(JOp)
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
    return J
end

struct HessianOperator{F, A}
    J::JacobianOperator{F, A}
    J_cache::JacobianOperator{F, A}
end
HessianOperator(J) = HessianOperator(J, Enzyme.make_zero(J))

Base.size(H::HessianOperator) = size(H.J)
Base.eltype(H::HessianOperator) = eltype(H.J)

function mul!(out, H::HessianOperator, v)
    _out = similar(H.J.res) # TODO cache in H
    du = Enzyme.make_zero(H.J.u) # TODO cache in H

    autodiff(Forward, mul!, 
        DuplicatedNoNeed(_out, out), 
        DuplicatedNoNeed(H.J, H.J_cache), 
        DuplicatedNoNeed(du, v))

    return nothing
end

##
# Newton-Krylov
##
import Base: @kwdef

"""
    Forcing

Implements forcing for inexact Newton-Krylov.
The equation ``‖F′(u)d + F(u)‖ <= η * ‖F(u)‖`` gives
the inexact Newton termination criterion.

## Implemented variants
- [`Fixed`](@ref)
- [`EisenstatWalker`](@ref)
"""
abstract type Forcing end

"""
    Fixed(η = 0.1)
"""
@kwdef struct Fixed <: Forcing
    η::Float64 = 0.1
end

function (F::Fixed)(args...)
    return F.η
end
inital(F::Fixed) = F.η

"""
    EisenstatWalker(η_max = 0.999, γ = 0.9)
"""
@kwdef struct EisenstatWalker <: Forcing
    η_max::Float64 = 0.999
    γ::Float64 = 0.9
end

# @assert η_max === nothing || 0.0 < η_max < 1.0

"""
Compute the Eisenstat-Walker forcing term for n > 0
"""
function (F::EisenstatWalker)(η, tol, n_res, n_res_prior)
    η_res = F.γ * n_res^2 / n_res_prior^2
    # Eq 3.6
    if F.γ * η^2 <= 1 // 10
        η_safe = min(F.η_max, η_res)
    else
        η_safe = min(F.η_max, max(η_res, F.γ * η^2))
    end
    return min(F.η_max, max(η_safe, 1 // 2 * tol / n_res)) # Eq 3.5
end
inital(F::EisenstatWalker) = F.η_max

const KWARGS_DOCS = """
## Keyword Arguments
  - `tol_rel`: Relative tolerance
  - `tol_abs`: Absolute tolerance
  - `max_niter`: Maximum number of iterations
  - `forcing`: Maximum forcing term for inexact Newton.
             If `nothing` an exact Newton method is used.  
  - `verbose`:
  - `Solver`:
  - `M`:
  - `N`:
  - `krylov_kwarg`
  - `callback`:
"""

"""
    newton_krylov(F, u₀::AbstractArray, M::Int = length(u₀); kwargs...)

## Arguments
  - `F`: `res = F(u₀)` solves `res = F(u₀) = 0`
  - `u₀`: Initial guess
  - `M`: Length of the output of `F`. Defaults to `length(u₀)`.
  
$(KWARGS_DOCS)
"""
function newton_krylov(F, u₀::AbstractArray, M::Int = length(u₀); kwargs...)
    F!(res, u) = (res .= F(u); nothing)
    return newton_krylov!(F!, u₀, M; kwargs...)
end

"""
## Arguments
  - `F!`: `F!(res, u)` solves `res = F(u) = 0`
  - `u₀`: Initial guess
  - `M`: Length of  the output of `F!`. Defaults to `length(u₀)`

$(KWARGS_DOCS)
"""
function newton_krylov!(F!, u₀::AbstractArray, M::Int = length(u₀); kwargs...)
    res = similar(u₀, M)
    return newton_krylov!(F!, u₀, res; kwargs...)
end

struct Stats
    outer_iterations::Int
    inner_iterations::Int
end
function update(stats::Stats, inner_iterations)
    return Stats(
        stats.outer_iterations + 1,
        stats.inner_iterations + inner_iterations
    )
end

"""

## Arguments
  - `F!`: `F!(res, u)` solves `res = F(u) = 0`
  - `u`: Initial guess
  - `res`: Temporary for residual
 
$(KWARGS_DOCS)
"""
function newton_krylov!(
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

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" Solver res₀ = n_res tol tol_rel tol_abs η

    J = JacobianOperator(F!, res, u)
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

        # Solve: Jx = -res
        # res is modifyed by J, so we create a copy `-res`
        # TODO: provide a temporary storage for `-res`
        solve!(solver, J, -res; kwargs...)

        d = solver.x # Newton direction
        s = 1        # Newton step TODO: LineSearch

        # Update u
        u .+= s .* d

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

        verbose > 0 && @info "Newton" iter = n_res η stats
        stats = update(stats, solver.stats.niter)
    end
    t = (time_ns() - t₀) / 1.0e9
    return u, (; solved = n_res <= tol, stats, t)
end

end # module NewtonKrylov
