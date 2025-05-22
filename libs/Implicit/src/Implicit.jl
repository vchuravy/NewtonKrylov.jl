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
abstract type SimpleImplicitAlgorithm end

struct ImplicitEuler <: SimpleImplicitAlgorithm end
function (::ImplicitEuler)(res, uₙ, Δt, f!, du, u, p, t)
    f!(du, u, p, t)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

struct ImplicitMidpoint <: SimpleImplicitAlgorithm end
function (::ImplicitMidpoint)(res, uₙ, Δt, f!, du, u, p, t; α = 0.5)
    ## Use res for a temporary allocation (uₙ .+ u) ./ 2
    uuₙ = res
    uuₙ .= (α .* uₙ .+ (1 - α) .* u)
    f!(du, uuₙ, p, t + α * Δt)

    res .= uₙ .+ Δt .* du .- u
    return nothing
end

struct ImplicitTrapezoid <: SimpleImplicitAlgorithm end
function (::ImplicitTrapezoid)(res, uₙ, Δt, f!, du, u, p, t)
    ## Use res as the temporary
    duₙ = res
    f!(duₙ, uₙ, p, t)
    f!(du, u, p, t + Δt)

    res .= uₙ .+ (Δt / 2) .* (duₙ .+ du) .- u
    return nothing
end

function nonlinear_problem(alg::SimpleImplicitAlgorithm, f::F) where {F}
    return (res, u, (uₙ, Δt, du, p, t)) -> alg(res, uₙ, Δt, f, du, u, p, t)
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
        RealT <: Real, uType, Params, Sol, F, Alg,
        SimpleImplicitOptions,
    } <: AbstractTimeIntegrator
    u::uType
    du::uType
    u_tmp::uType
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
        ode::ODEProblem, alg::SimpleImplicitAlgorithm;
        dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...
    )
    u = copy(ode.u0)
    du = zero(u)
    res = zero(u)
    u_tmp = similar(u)
    t = first(ode.tspan)
    iter = 0
    integrator = SimpleImplicit(
        u, du, u_tmp, res, t, dt, zero(dt), iter, ode.p,
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
    integrator.u_tmp .= integrator.u

    F! = nonlinear_problem(alg, integrator.f)
    _, stats = NewtonKrylov.newton_krylov!(
        F!, integrator.u_tmp, (integrator.u, integrator.dt, integrator.du, integrator.p, integrator.t), integrator.res;
        verbose = integrator.opts.verbose, krylov_kwargs = integrator.opts.krylov_kwargs,
        algo = integrator.opts.algo, tol_abs = 6.0e-6
    )
    @assert stats.solved
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
