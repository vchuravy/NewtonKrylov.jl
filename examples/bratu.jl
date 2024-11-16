# 1D bratu equation

using NewtonKrylov, Krylov
using SparseArrays, LinearAlgebra

function bratu!(res, y, Δx, λ)
	N = length(y)
	for i in 1:N
		y_l = i == 1 ? zero(eltype(y)) : y[i - 1]
		y_r = i == N ? zero(eltype(y)) : y[i + 1]
		y′′ = (y_r - 2y[i] + y_l) / Δx^2

		res[i] = y′′ + λ * exp(y[i]) # = 0
	end
	nothing
end

function bratu(y, dx, λ)
	res = similar(y)
	bratu!(res, y, dx, λ)
	res
end

function true_sol_bratu(x)
	# for λ = 3.51382, 2nd sol θ = 4.8057
	θ = 4.79173
	-2 * log(cosh(θ * (x-0.5)/2) / (cosh(θ/4)))
end

const N = 10_000
const λ = 3.51382
const dx = 1 / (N + 1) # Grid-spacing

x  = LinRange(0.0+dx, 1.0 - dx, N)
u₀ = sin.(x.* π)

## Build the Jacobian once to inspect it
function assemble_jacobian(JOp)
	v = zeros(eltype(JOp), size(JOp)[2])
	out = zeros(eltype(JOp), size(JOp)[1])
    J = SparseMatrixCSC{eltype(v), Int}(undef, size(JOp)...)
    for i in 1:N
        out .= 0.0
        v .= 0.0
        v[i] = 1.0
        mul!(out, JOp, v)
        for j in 1:N
            if out[j] != 0
                J[i, j] = out[j] # TODO: Check i,j
            end
        end
    end
    J
end

JOp = NewtonKrylov.JacobianOperator((res, u) -> bratu!(res, u, dx, λ), similar(u₀), copy(u₀))
J = assemble_jacobian(JOp)
J2 = assemble_jacobian(adjoint(JOp))
J == J2 # since J is symmetric

reference = true_sol_bratu.(x)
uₖ_1 = newton_krylov!(
	(res, u) -> bratu!(res, u, dx, λ),
	copy(u₀), similar(u₀);
	verbose = 1
)

uₖ_2 = newton_krylov(
	(u) -> bratu(u, dx, λ),
	copy(u₀);
	verbose = 1
)

ϵ1 = abs2.(uₖ_1 .- reference)
ϵ2 = abs2.(uₖ_1 .- reference)

# Very slow?
# newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	verbose = true,
# 	Solver = GmresSolver
# )

# Explodes..
newton_krylov!(
	(res, u) -> bratu!(res, u, dx, λ),
	copy(u₀), similar(u₀);
	verbose = 1,
	Solver = CglsSolver, # CgneSolver
)

newton_krylov!(
	(res, u) -> bratu!(res, u, dx, λ),
	copy(u₀), similar(u₀);
	verbose = 1,
	Solver = BicgstabSolver,
	η_max = nothing
)