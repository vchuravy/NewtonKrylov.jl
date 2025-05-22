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

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(
    semi, interval = analysis_interval,
    extra_analysis_integrals = (entropy, energy_total)
)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(
    interval = 100,
    save_final_restart = true
)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(
    interval = 100,
    save_initial_solution = true,
    save_final_solution = true,
    solution_variables = cons2prim
)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(
    summary_callback,
    analysis_callback, alive_callback,
    save_restart, save_solution,
    stepsize_callback
)

###############################################################################
# run the simulation

using NewtonKrylov
using Implicit

# ## Jacobian
J = Implicit.jacobian(Implicit.ImplicitEuler(), ode.f, ode.u0, ode.p, 0.1, first(ode.tspan))

# ## Solve with explicit timesteps

Δt = 0.01
ts = first(ode.tspan):Δt:last(ode.tspan)
Simple.solve(Simple.G_Euler!, ode.f, ode.u0, ode.p, Δt, ts; verbose = 1, krylov_kwargs = (; verbose = 1))

# ## Solve using ODE interface

sol = solve(
    ode, Implicit.ImplicitEuler();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks
);
