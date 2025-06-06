module Ariadne

export newton_krylov, newton_krylov!

using Krylov
using LinearAlgebra, SparseArrays
using Enzyme

##
# JacobianOperator
##
import LinearAlgebra: mul!

function maybe_duplicated(x, ::Val{N} = Val(1)) where {N}
    # TODO cache?
    if !Enzyme.Compiler.guaranteed_const(typeof(x))
        if N == 1
            return Duplicated(x, Enzyme.make_zero(x))
        else
            return BatchDuplicated(x, ntuple(_ -> Enzyme.make_zero(x), Val(N)))
        end
    else
        return Const(x)
    end
end

# TODO: JacobianOperator with thunk

"""
    JacobianOperator

Efficient implementation of `J(f,x,p) * v` and `v * J(f, x,p)'`
"""
struct JacobianOperator{F, A, P}
    f::F # F!(res, u, p)
    res::A
    u::A
    p::P
    function JacobianOperator(f::F, res, u, p) where {F}
        return new{F, typeof(u), typeof(p)}(f, res, u, p)
    end
end

Base.size(J::JacobianOperator) = (length(J.res), length(J.u))
Base.eltype(J::JacobianOperator) = eltype(J.u)
Base.length(J::JacobianOperator) = prod(size(J))

function mul!(out::AbstractVector, J::JacobianOperator, v::AbstractVector)
    autodiff(
        Forward,
        maybe_duplicated(J.f), Const,
        Duplicated(J.res, reshape(out, size(J.res))),
        Duplicated(J.u, reshape(v, size(J.u))),
        maybe_duplicated(J.p)
    )
    return nothing
end

if VERSION >= v"1.11.0"

    function tuple_of_vectors(M::Matrix{T}, shape) where {T}
        n, m = size(M)
        return ntuple(m) do i
            vec = Base.wrap(Array, memoryref(M.ref, (i - 1) * n + 1), (n,))
            reshape(vec, shape)
        end
    end

    function mul!(Out::AbstractMatrix, J::JacobianOperator, V::AbstractMatrix)
        @assert size(Out, 2) == size(V, 2)
        out = tuple_of_vectors(Out, size(J.res))
        v = tuple_of_vectors(V, size(J.u))

        N = length(out)
        autodiff(
            Forward,
            maybe_duplicated(J.f, Val(N)), Const,
            BatchDuplicated(J.res, out),
            BatchDuplicated(J.u, v),
            maybe_duplicated(J.p, Val(N))
        )
        return nothing
    end

end # VERSION >= v"1.11.0"

LinearAlgebra.adjoint(J::JacobianOperator) = Adjoint(J)
LinearAlgebra.transpose(J::JacobianOperator) = Transpose(J)

# Jᵀ(y, u) = ForwardDiff.gradient!(y, x -> dot(F(x), u), xk)
# or just reverse mode

function mul!(out::AbstractVector, J′::Union{Adjoint{<:Any, <:JacobianOperator}, Transpose{<:Any, <:JacobianOperator}}, v::AbstractVector)
    J = parent(J′)
    # TODO: provide cache for `copy(v)`
    # Enzyme zeros input derivatives and that confuses the solvers.
    # If `out` is non-zero we might get spurious gradients
    fill!(out, 0)
    autodiff(
        Reverse,
        maybe_duplicated(J.f), Const,
        Duplicated(J.res, reshape(copy(v), size(J.res))),
        Duplicated(J.u, reshape(out, size(J.u))),
        maybe_duplicated(J.p)
    )
    return nothing
end

if VERSION >= v"1.11.0"

    function mul!(Out::AbstractMatrix, J′::Union{Adjoint{<:Any, <:JacobianOperator}, Transpose{<:Any, <:JacobianOperator}}, V::AbstractMatrix)
        J = parent(J′)
        @assert size(Out, 2) == size(V, 2)

        # If `out` is non-zero we might get spurious gradients
        fill!(Out, 0)

        # TODO: provide cache for `copy(v)`
        # Enzyme zeros input derivatives and that confuses the solvers.
        V = copy(V)

        out = tuple_of_vectors(Out, size(J.u))
        v = tuple_of_vectors(V, size(J.res))

        N = length(out)

        # TODO: BatchDuplicated for J.f
        autodiff(
            Reverse,
            maybe_duplicated(J.f, Val(N)), Const,
            BatchDuplicated(J.res, v),
            BatchDuplicated(J.u, out),
            maybe_duplicated(J.p, Val(N))
        )
        return nothing
    end

end # VERSION >= v"1.11.0"

function Base.collect(JOp::Union{Adjoint{<:Any, <:JacobianOperator}, Transpose{<:Any, <:JacobianOperator}, JacobianOperator})
    N, M = size(JOp)
    if JOp isa JacobianOperator
        v = zero(JOp.u)
        out = zero(JOp.res)
    else
        v = zero(parent(JOp).res)
        out = zero(parent(JOp).u)
    end
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
  - `Workspace`:
  - `M`:
  - `N`:
  - `krylov_kwarg`
  - `callback`:
"""

"""
    newton_krylov(F, u₀::AbstractArray, M::Int = length(u₀); kwargs...)

## Arguments
  - `F`: `res = F(u₀, p)` solves `res = F(u₀) = 0`
  - `u₀`: Initial guess
  - `p`: Parameters
  - `M`: Length of the output of `F`. Defaults to `length(u₀)`.
  
$(KWARGS_DOCS)
"""
function newton_krylov(F, u₀::AbstractArray, p = nothing, M::Int = length(u₀); kwargs...)
    F!(res, u, p) = (res .= F(u, p); nothing)
    return newton_krylov!(F!, u₀, p, M; kwargs...)
end

"""
## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u₀`: Initial guess
  - `p`: Parameters
  - `M`: Length of  the output of `F!`. Defaults to `length(u₀)`

$(KWARGS_DOCS)
"""
function newton_krylov!(F!, u₀::AbstractArray, p = nothing, M::Int = length(u₀); kwargs...)
    res = similar(u₀, M)
    Enzyme.make_zero!(res) # u₀ .= 0 might ignore ghost cells
    return newton_krylov!(F!, u₀, p, res; kwargs...)
end

struct Stats
    outer_iterations::Int
    inner_iterations::Int
    n_res::Float64
end
function update(stats::Stats, inner_iterations, n_res::Float64)
    return Stats(
        stats.outer_iterations + 1,
        stats.inner_iterations + inner_iterations,
        n_res
    )
end

"""

## Arguments
  - `F!`: `F!(res, u, p)` solves `res = F(u) = 0`
  - `u`: Initial guess
  - `p`: 
  - `res`: Temporary for residual
 
$(KWARGS_DOCS)
"""
function newton_krylov!(
        F!, u::AbstractArray, p, res::AbstractArray;
        tol_rel = 1.0e-6,
        tol_abs = 1.0e-12, # Scipy uses 6e-6
        max_niter = 50,
        forcing::Union{Forcing, Nothing} = EisenstatWalker(),
        verbose = 0,
        algo = :gmres,
        M = nothing,
        N = nothing,
        krylov_kwargs = (;),
        callback = (args...) -> nothing,
    )
    t₀ = time_ns()
    F!(res, u, p) # res = F(u)
    n_res = norm(res)
    callback(u, res, n_res)

    tol = tol_rel * n_res + tol_abs

    if forcing !== nothing
        η = inital(forcing)
    end

    verbose > 0 && @info "Jacobian-Free Newton-Krylov" algo res₀ = n_res tol tol_rel tol_abs η

    J = JacobianOperator(F!, res, u, p)

    # TODO: Refactor to provide method that re-uses the cache here.
    kc = KrylovConstructor(res)
    workspace = krylov_workspace(algo, kc)

    stats = Stats(0, 0, n_res)
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
        krylov_solve!(workspace, J, copy(res); kwargs...)

        d = workspace.x # Newton direction
        s = 1        # Newton step TODO: LineSearch

        # Update u
        u .-= s .* d

        # Update residual and norm
        n_res_prior = n_res

        F!(res, u, p) # res = F(u)
        n_res = norm(res)
        callback(u, res, n_res)

        if isinf(n_res) || isnan(n_res)
            @error "Inner solver blew up" stats
            break
        end

        if forcing !== nothing
            η = forcing(η, tol, n_res, n_res_prior)
        end

        # This is almost to be expected for implicit time-stepping
        if verbose > 0 && workspace.stats.niter == 0 && forcing !== nothing
            @info "Inexact Newton thinks our step is good enough " η stats
        end

        stats = update(stats, workspace.stats.niter, n_res)
        verbose > 0 && @info "Newton" iter = n_res η stats
    end
    t = (time_ns() - t₀) / 1.0e9
    return u, (; solved = n_res <= tol, stats, t)
end

end # module Ariadne
