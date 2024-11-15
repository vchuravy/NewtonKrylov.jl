# 1D bratu equation

using NewtonKrylov

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

J! = JacobianOperatorInPlace{N}(
	(res, u) -> bratu!(res, u, dx, λ),
	copy(u₀)
)

J = JacobianOperator{N}(
	(u) -> bratu(u, dx, λ),
	copy(u₀)
)

size(J!)
eltype(J!)
reference = true_sol_bratu.(x)
solution  = newton_krylov!(J!)
J!.u .= u₀
@time newton_krylov!(J!)
ϵ = abs2.(solution .- reference)

solution  = newton_krylov!(J)
J.u .= u₀
@time newton_krylov!(J)
ϵ2 = abs2.(solution .- reference)
