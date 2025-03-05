using Trixi

# Example based on https://github.com/trixi-framework/Trixi.jl/blob/main/examples/tree_1d_dgsem/elixir_advection_extended.jl

###############################################################################
# semidiscretization of the linear advection diffusion equation

advection_velocity = 0.1
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.1
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -convert(Float64, pi) # minimum coordinate
coordinates_max = convert(Float64, pi) # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(
    coordinates_min, coordinates_max,
    initial_refinement_level = 4,
    n_cells_max = 30_000, # set maximum capacity of tree data structure
    periodicity = true
)

function x_trans_periodic(
        x, domain_length = SVector(oftype(x[1], 2 * pi)),
        center = SVector(oftype(x[1], 0))
    )
    x_normalized = x .- center
    x_shifted = x_normalized .% domain_length
    x_offset = (
        (x_shifted .< -0.5f0 * domain_length) -
            (x_shifted .> 0.5f0 * domain_length)
    ) .*
        domain_length
    return center + x_shifted + x_offset
end

# Define initial condition
function initial_condition_diffusive_convergence_test(
        x, t,
        equation::LinearScalarAdvectionEquation1D
    )
    # Store translated coordinate for easy use of exact solution
    x_trans = x_trans_periodic(x - equation.advection_velocity * t)

    nu = diffusivity()
    c = 0
    A = 1
    omega = 1
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(
    mesh, (equations, equations_parabolic),
    initial_condition,
    solver;
    boundary_conditions = (
        boundary_conditions,
        boundary_conditions_parabolic,
    )
)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

##
# Verify the gradient will work
##

using Enzyme
using DiffEqBase

prob = ode

u0 = prob.u0
p = prob.p
tmp2 = Enzyme.make_zero(p)
t = prob.tspan[1]
du = zero(u0)

if DiffEqBase.isinplace(prob)
    _f = prob.f
else
    _f = (du, u, p, t) -> (du .= prob.f(u, p, t); nothing)
end

_tmp6 = Enzyme.make_zero(_f)
tmp3 = zero(u0)
tmp4 = zero(u0)
ytmp = zero(u0)
tmp1 = zero(u0)

Enzyme.autodiff(
    Enzyme.Reverse, Enzyme.Duplicated(_f, _tmp6),
    Enzyme.Const, Enzyme.Duplicated(tmp3, tmp4),
    Enzyme.Duplicated(ytmp, tmp1),
    Enzyme.Duplicated(p, tmp2),
    Enzyme.Const(t)
)

using NewtonKrylov

function G_Euler!(res, f!, du, u, uₙ, t, Δt, p)
    f!(du, u, p, t)
    return res .= uₙ .+ Δt .* du .- u
end

function solve!(ode)
    if DiffEqBase.isinplace(ode)
        f = ode.f
    else
        f = (du, u, p, t) -> (du .= ode.f(u, p, t); nothing)
    end

    Δt = 0.1
    uₙ = ode.u0
    t = first(ode.tspan)
    du = similar(ode.u0)
    res = similar(ode.u0)

    F!(res, u) = G_Euler!(res, f, du, u, uₙ, t, Δt, p)

    while t <= last(ode.tspan)
        u, stats = newton_krylov!(F!, copy(uₙ), res)
        @show stats
        uₙ .= u
    end
    return uₙ
end

solve!(ode)
