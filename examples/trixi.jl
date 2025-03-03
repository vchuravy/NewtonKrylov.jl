# using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Semidiscretization of the quasi 1d shallow water equations
# See Chan et al.  https://doi.org/10.48550/arXiv.2307.12089 for details

equations = ShallowWaterEquationsQuasi1D(gravity_constant = 9.81)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_chan_etal, flux_nonconservative_chan_etal)
surface_flux = (FluxPlusDissipation(flux_chan_etal, DissipationLocalLaxFriedrichs()),
                flux_nonconservative_chan_etal)

dg = DGMulti(polydeg = 4, element_type = Line(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

cells_per_dimension = (8,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (0.0,), coordinates_max = (sqrt(2),),
                   periodicity = true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

# sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-8, reltol = 1.0e-8,
#             ode_default_options()..., callback = callbacks)

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

Enzyme.autodiff(Enzyme.Reverse, Enzyme.Duplicated(_f, _tmp6),
    Enzyme.Const, Enzyme.Duplicated(tmp3, tmp4),
    Enzyme.Duplicated(ytmp, tmp1),
    Enzyme.Duplicated(p, tmp2),
    Enzyme.Const(t))
