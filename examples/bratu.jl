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

reference = true_sol_bratu.(x)
uₖ_1 = newton_krylov!(
	(res, u) -> bratu!(res, u, dx, λ),
	copy(u₀), similar(u₀);
	verbose = true
)

uₖ_2 = newton_krylov(
	(u) -> bratu(u, dx, λ),
	copy(u₀);
	verbose = true
)

ϵ1 = abs2.(uₖ_1 .- reference)
ϵ2 = abs2.(uₖ_1 .- reference)

# Very slow?
# newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	verbose = true,
# 	solver = :gmres
# )

# Explodes..
# newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	verbose = true,
# 	solver = :cgne
# 	η_max = nothing
# )

# newton_krylov!(
# 	(res, u) -> bratu!(res, u, dx, λ),
# 	copy(u₀), similar(u₀);
# 	verbose = true,
# 	solver = :bicgstab,
# 	η_max = nothing
# )