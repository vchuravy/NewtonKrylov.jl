# 1D bratu equation

using NewtonKrylov, Krylov
using KrylovPreconditioners
using SparseArrays, LinearAlgebra

function bratu!(res, y, Δx, λ)
    N = length(y)
    for i in 1:N
        y_l = i == 1 ? zero(eltype(y)) : y[i - 1]
        y_r = i == N ? zero(eltype(y)) : y[i + 1]
        y′′ = (y_r - 2y[i] + y_l) / Δx^2

        res[i] = y′′ + λ * exp(y[i]) # = 0
    end
    return nothing
end

function bratu(y, dx, λ)
    res = similar(y)
    bratu!(res, y, dx, λ)
    return res
end

function true_sol_bratu(x)
    # for λ = 3.51382, 2nd sol θ = 4.8057
    θ = 4.79173
    return -2 * log(cosh(θ * (x - 0.5) / 2) / (cosh(θ / 4)))
end

const N = 10_000
const λ = 3.51382
const dx = 1 / (N + 1) # Grid-spacing

x = LinRange(0.0 + dx, 1.0 - dx, N)
u₀ = sin.(x .* π)

reference = true_sol_bratu.(x)

uₖ_1 = newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ),
    copy(u₀), similar(u₀);
    Solver = CgSolver,
)

uₖ_2 = newton_krylov(
    (u) -> bratu(u, dx, λ),
    copy(u₀);
    Solver = CgSolver
)

ϵ1 = abs2.(uₖ_1 .- reference)
ϵ2 = abs2.(uₖ_1 .- reference)

##
# Solving with a fixed forcing
newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ),
    copy(u₀), similar(u₀);
    Solver = CgSolver,
    forcing = NewtonKrylov.Fixed()
)

##
# Solving with no forcing
newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ),
    copy(u₀), similar(u₀);
    Solver = CgSolver,
    forcing = nothing
)

##
# Solve using GMRES -- very slow
# @time newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	Solver = GmresSolver,
# )

##
# Solve using GMRES + ILU Preconditoner
@time newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ),
    copy(u₀), similar(u₀);
    Solver = GmresSolver,
    N = (J) -> ilu(collect(J)), # Assembles the full Jacobian
    ldiv = true,
)

##
# Solve using FGMRES + ILU Preconditoner
@time newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ),
    copy(u₀), similar(u₀);
    Solver = FgmresSolver,
    N = (J) -> ilu(collect(J)), # Assembles the full Jacobian
    ldiv = true,
)

##
# Solve using FGMRES + GMRES Preconditoner
struct GmresPreconditioner{JOp}
    J::JOp
    itmax::Int
end

function LinearAlgebra.mul!(y, P::GmresPreconditioner, x)
    sol, _ = gmres(P.J, x; P.itmax)
    return copyto!(y, sol)
end

@time newton_krylov!(
    (res, u) -> bratu!(res, u, dx, λ),
    copy(u₀), similar(u₀);
    Solver = FgmresSolver,
    N = (J) -> GmresPreconditioner(J, 30),
)

# # Explodes..
# newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	verbose = 1,
# 	Solver = CglsSolver, # CgneSolver
# )

# newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	verbose = 1,
# 	Solver = BicgstabSolver,
# 	η_max = nothing
# )
