# # Using the an implicit solver based on Ariadne with Trixi.jl

using Trixi
using Implicit
using CairoMakie


# Notes:
# Must disable both Polyester and LoopVectorization for Enzyme to be able to differentiate Trixi.jl
# Using https://github.com/trixi-framework/Trixi.jl/pull/2295
#
# LocalPreferences.jl
# ```toml
# [Trixi]
# loop_vectorization = false
# polyester = false
# ```

@assert !Trixi._PREFERENCE_POLYESTER
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

# ## Load Trixi Example
# trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), sol = nothing);
trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"));

ref = copy(sol)

u = copy(ode.u0)
du = zero(ode.u0)
res = zero(ode.u0)

F! = Implicit.nonlinear_problem(Implicit.ImplicitEuler(), ode.f)
J = Implicit.Ariadne.JacobianOperator(F!, res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

collect(J)

using LinearAlgebra
out = zero(u)
v = zero(u)
@time mul!(u, J, v)
@time F!(res, u, (ode.u0, 1.0, du, ode.p, 0.0, (), 1))

F! = Implicit.nonlinear_problem(Implicit.SDIRK2(), ode.f)
u1 = copy(ode.u0)
u2 = copy(u1)
J1 = Implicit.Ariadne.JacobianOperator(F!, res, u1, (ode.u0, 1.0, du, ode.p, 0.0, (u1,), 1))
J2 = Implicit.Ariadne.JacobianOperator(F!, res, u2, (ode.u0, 1.0, du, ode.p, 0.0, (u1,), 2))

using LinearAlgebra
out = zero(u)
v = zero(u)
@time F!(res, u, (ode.u0, 1.0, du, ode.p, 0.0, (u1,), 1))
@time mul!(u, J1, v)

collect(J1)
collect(J2)

# Cost of time(Jvp) ≈ 2 * time(rhs)

# ### Jacobian (of the implicit function given the ode)
J = Implicit.jacobian(Implicit.ImplicitEuler(), ode, 1.0)

# ### Solve using ODE interface

sol = solve(
    ode, Implicit.ImplicitEuler();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

sol = solve(
    ode, Implicit.ImplicitMidpoint();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

sol = solve(
    ode, Implicit.ImplicitTrapezoid();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);


sol = solve(
    ode, Implicit.TRBDF2();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

import OrdinaryDiffEqSDIRK
import DifferentiationInterface: AutoFiniteDiff
sol = solve(
    ode, OrdinaryDiffEqSDIRK.TRBDF2(autodiff = AutoFiniteDiff());
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    adaptive = false
);

# ### Plot the (reference) solution

# We have to manually convert the sol since Implicit has it's own leightweight solution type.
# Create an extension.
## pd = PlotData2D(sol.u[end], sol.prob.p)

plot(Trixi.PlotData2DTriangulated(ref.u[1], ref.prob.p))

# ### Plot the solution

plot(Trixi.PlotData2DTriangulated(sol.u[end], sol.prob.p))

# ## Increase CFL numbers

trixi_include(joinpath(examples_dir(), "tree_2d_dgsem", "elixir_advection_basic.jl"), cfl = 10, sol = nothing);

sol = solve(
    ode, Implicit.ImplicitEuler();
    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
    ode_default_options()..., callback = callbacks,
    # verbose=1,
    krylov_algo = :gmres,
    # krylov_kwargs=(;verbose=1)
);

@show callbacks.discrete_callbacks[4]

# ### Plot the solution

plot(Trixi.PlotData2DTriangulated(sol.u[end], sol.prob.p))
