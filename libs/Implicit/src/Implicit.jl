module Implicit

using UnPack
import NewtonKrylov

# Wrapper type for solutions from Implicit.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

# Abstract supertype of Implict.jl's own time integrators for dispatch
abstract type AbstractTimeIntegrator end

import DiffEqBase

import DiffEqBase: solve, CallbackSet, ODEProblem
export solve, ODEProblem

# Interface required by DiffEqCallbacks.jl
function DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator)
    return integrator.opts.tstops
end
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
    return get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
    return maximum(get_tstops_array(integrator))
end

function finalize_callbacks(integrator::AbstractTimeIntegrator)
    callbacks = integrator.opts.callback

    return if callbacks isa CallbackSet
        foreach(callbacks.discrete_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
        foreach(callbacks.continuous_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
    end
end

import SciMLBase: get_du, get_tmp_cache, u_modified!,
    init, step!, check_error,
    get_proposed_dt, set_proposed_dt!,
    terminate!, remake, add_tstop!, has_tstop, first_tstop


# Abstract base type for time integration schemes
abstract type SimpleImplicitAlgorithm{N} end

stages(::SimpleImplicitAlgorithm{N}) where {N} = N

"""
```
1 | 1
--|--
  | 1
```
"""
struct ImplicitEuler <: SimpleImplicitAlgorithm{1} end
function (::ImplicitEuler)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    f!(du, u, p, t) # t = t0 + c_1 * Δt

    res .= uₙ .+ Δt .* du .- u # Δt * a_11
    return nothing
end

"""
```
1/2 | 1/2
----|----
    | 1
```
"""
struct ImplicitMidpoint <: SimpleImplicitAlgorithm{1} end
function (::ImplicitMidpoint)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    ## Use res for a temporary allocation (uₙ .+ u) ./ 2
    uuₙ = res
    uuₙ .= 0.5 .* (uₙ .+ u)
    f!(du, uuₙ, p, t + 0.5 * Δt)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

"""
```
0 |  0   0
1 | 1/2 1/2
--|--------
  | 1/2 1/2
```
"""
struct ImplicitTrapezoid <: SimpleImplicitAlgorithm{1} end
function (::ImplicitTrapezoid)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    ## Use res as the temporary
    duₙ = res
    f!(duₙ, uₙ, p, t)
    f!(du, u, p, t + Δt)

    res .= uₙ .+ (Δt / 2) .* (duₙ .+ du) .- u
    return nothing
end

struct TRBDF2 <: SimpleImplicitAlgorithm{2} end
function (::TRBDF2)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
    γ = 2 - √2
    return if stage == 1
        # Stage 1: Trapezoidal to t + γΔt
        duₙ = res
        f!(duₙ, uₙ, p, t)
        f!(du, u, p, t + γ * Δt)

        res .= uₙ .+ (γ * Δt / 2) .* (duₙ .+ du) .- u
    else
        u₁ = stages[1]
        f!(du, u, p, t + Δt)

        # BDF2 coefficients
        c₁ = 1 / γ^2
        c₂ = -(1 - γ)^2 / γ^2
        c₃ = (1 - γ)^2 * Δt / (γ * (2 - γ))

        res .= c₁ .* u₁ .+ c₂ .* uₙ .+ c₃ .* du .- u
    end
end

# struct ESDIRK2 <: SimpleImplicitAlgorithm{2} end
# function (::ESDIRK2)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
#     # Standard ESDIRK2 parameters
#     γ = 1 - 1/√2  # ≈ 0.2928932188134525
#     a₂₁ = 1 - γ   # ≈ 0.7071067811865476
#     # # Alternative parameter set
#     # γ = (2 + √2) / 4  # ≈ 0.8535533905932738
#     # a₂₁ = (2 - √2) / 4  # ≈ 0.1464466094067262

#     if stage == 1
#         # Stage 1: Explicit stage
#         # k₁ = f(tₙ, uₙ)
#         # u₁ = uₙ + γΔt * k₁
#         # This is explicit, so no nonlinear solve needed
#         f!(du, uₙ, p, t)
#         res .= uₙ .+ γ * Δt .* du .- u
#     else
#         # Stage 2: Implicit stage
#         # k₂ = f(tₙ + γΔt, u₁) (from stage 1)
#         # u₂ = uₙ + Δt(a₂₁ * k₁ + γ * k₂)
#         #    = uₙ + Δt((1-γ) * k₁ + γ * k₂)

#         u₁ = stages[1]

#         # Get k₁ = (u₁ - uₙ) / (γΔt) from the explicit stage 1 result
#         k₁ = res
#         k₁ .= (u₁ .- uₙ) ./ (γ * Δt)

#         # k₂ = f(tₙ + γΔt, u₂) - this is what we're solving for
#         f!(du, u, p, t + γ * Δt)

#         res .= uₙ .+ Δt .* (a₂₁ .* k₁ .+ γ .* du) .- u
#     end
# end

# """
# SDIRK2 (γ = (2-√2)/2 ≈ 0.293)
# ```
# ```
# γ   |  γ   0
# 1-γ | 1-2γ γ
# ----|--------
#     | 1/2 1/2
# ```
# """
# struct SDIRK2 <: SimpleImplicitAlgorithm{2} end

# function (::SDIRK2)(res, uₙ, Δt, f!, du, u, p, t, stages, stage)
#     γ = (2 + √2) / 4  # ≈ 0.8535533905932738
#     a₂₁ = (2 - √2) / 4  # ≈ 0.1464466094067262

#     if stage == 1
#         # Stage 1: Implicit
#         f!(du, u, p, t + γ * Δt)
#         res .= uₙ .+ γ * Δt .* du .- u
#     elseif stage == 2
#         # Stage 2: Implicit
#         u₁ = stages[1]
#         k₁ = res
#         k₁ .= (u₁ .- uₙ) ./ (γ * Δt)

#         # Compute f(u₂, t + (1-γ)*Δt)
#         f!(du, u, p, t + (1 - γ) * Δt)

#         res .= uₙ .+ Δt .* (a₂₁ .* k₁ .+ γ .* du) .- u
#     end
#     return nothing
# end

function nonlinear_problem(alg::SimpleImplicitAlgorithm, f::F) where {F}
    return (res, u, (uₙ, Δt, du, p, t, stages, stage)) -> alg(res, uₙ, Δt, f, du, u, p, t, stages, stage)
end


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleImplicitOptions{Callback}
    callback::Callback # callbacks; used in Trixi.jl
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal number of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
    verbose::Int
    algo::Symbol
    krylov_kwargs::Any
end


function SimpleImplicitOptions(callback, tspan; maxiters = typemax(Int), verbose = 0, krylov_algo = :gmres, krylov_kwargs = (;), kwargs...)
    return SimpleImplicitOptions{typeof(callback)}(
        callback, false, Inf, maxiters,
        [last(tspan)],
        verbose,
        krylov_algo,
        krylov_kwargs
    )
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct SimpleImplicit{
        RealT <: Real, uType, Params, Sol, F, M, Alg <: SimpleImplicitAlgorithm,
        SimpleImplicitOptions,
    } <: AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
    stages::NTuple{M, uType}
    res::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # ignored
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi.jl
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::Alg # SimpleImplicitAlgorithm
    opts::SimpleImplicitOptions
    finalstep::Bool # added for convenience
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleImplicit, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(
        ode::ODEProblem, alg::SimpleImplicitAlgorithm{N};
        dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...
    ) where {N}
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    stages = ntuple(_ -> similar(u), Val(N - 1))
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleImplicit(
        u, du, u_tmp, stages, res, t, dt, zero(dt), iter, ode.p,
        (prob = ode,), ode.f, alg,
        SimpleImplicitOptions(
            callback, ode.tspan;
            kwargs...
        ), false
    )

    # initialize callbacks
    if callback isa CallbackSet
        foreach(callback.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported with the implicit time integration methods."))
        end
        foreach(callback.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(
        ode::ODEProblem, alg::SimpleImplicitAlgorithm;
        dt, callback = nothing, kwargs...
    )
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    return solve!(integrator)
end

function solve!(integrator::SimpleImplicit)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    finalize_callbacks(integrator)

    return TimeIntegratorSolution(
        (first(prob.tspan), integrator.t),
        (prob.u0, integrator.u),
        integrator.sol.prob
    )
end

function step!(integrator::SimpleImplicit)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
            isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    # one time step
    integrator.u_tmp
    for stage in 1:stages(alg)
        F! = nonlinear_problem(alg, integrator.f)
        # TODO: Pass in `stages[1:(stage-1)]` or full tuple?
        _, stats = NewtonKrylov.newton_krylov!(
            F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.p, integrator.t, integrator.stages, stage), integrator.res;
            verbose = integrator.opts.verbose, krylov_kwargs = integrator.opts.krylov_kwargs,
            algo = integrator.opts.algo, tol_abs = 6.0e-6
        )
        @assert stats.solved
        if stage < stages(alg)
            integrator.stages[stage] .= integrator.u_tmp
        end
    end
    integrator.u .= integrator.u_tmp

    integrator.iter += 1
    integrator.t += integrator.dt

    begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    return if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleImplicit) = integrator.du
get_tmp_cache(integrator::SimpleImplicit) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleImplicit, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleImplicit, dt)
    return integrator.dt = dt
end

# Required e.g. for `glm_speed_callback`
function get_proposed_dt(integrator::SimpleImplicit)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::SimpleImplicit)
    integrator.finalstep = true
    return empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleImplicit, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    return resize!(integrator.u_tmp, new_size)
end

### Helper
jacobian(G!, ode::ODEProblem, Δt) = jacobian(G!, ode.f, ode.u0, ode.p, Δt, first(ode.tspan))

function jacobian(G!, f!, uₙ, p, Δt, t)
    u = copy(uₙ)
    du = zero(uₙ)
    res = zero(uₙ)

    F! = nonlinear_problem(G!, f!)

    J = NewtonKrylov.JacobianOperator(F!, res, u, (uₙ, Δt, du, p, t))
    return collect(J)
end

end # module Implicit
