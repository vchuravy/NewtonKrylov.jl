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

using NewtonKrylov

include(joinpath(dirname(pathof(NewtonKrylov)), "..", "examples", "implicit.jl"))

# ## Jacobian
J = jacobian(G_Euler!, ode.f, ode.u0, ode.p, 0.1, first(ode.tspan))

# ## Solve with fixed timestep

Δt = 0.01
ts = first(ode.tspan):Δt:last(ode.tspan)
solve(G_Euler!, ode.f, ode.u0, ode.p, Δt, ts; verbose = 1, krylov_kwargs = (; verbose = 1))
